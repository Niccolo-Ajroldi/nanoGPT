from collections import defaultdict
from scipy import stats
import math
import torch
import torch.nn.functional as F
import wandb

class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 
               2 - keep most recent 1/2, 
               3 - keep most recent 1/3,
               ... 
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        '''
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0            # start and end are pointer to the memory chunck
        self.total = 0          # total number of elements added (including leaked)
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total = 0

    def double_size(self):
        self.size *= 2
        self.buffer.resize_(self.size)

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            self.double_size()                      # double the size of buffer

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total % self.ratio == 0:    # if leaky_count is multiple of ratio
                self.count += 1                     # increase count in queue by one
            else:                               # otherwise leak and keep same count
                self.start += 1                     # increase start index by one

        self.total += 1                         # always increase total count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0:self.count] = self.buffer[self.start:self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), 
                               kernel_size=b_n, stride=b_n).view(-1)
            centered = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(centered).item()
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), 
                               kernel_size=b_n, stride=1).view(-1)
            centered = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(centered).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof

class HGWarmup(object):
    
    def __init__(self, optimizer, lr,
                 adapt_period=100, leak_ratio=1, adapt_alpha='dot', alpha_factor=2., 
                 conf_level=0.95, var_mode='bm', max_lr = 10.0,
                 stopping="first", increase_crit="great",
                 ):
        
        # if adapt_period < 100:
        #     raise ValueError("Period for statistics may be too small: {}".format(adapt_period))
        if leak_ratio < 1:
            raise ValueError("Invalid value for leak_ratio (>=1): {}".format(leak_ratio))
        if adapt_alpha not in ('dot', 'cos'):
            raise ValueError("adapt_alpha not in ('dot', 'cos'){}".format(adapt_alpha))
        if alpha_factor <= 1.0:
            raise ValueError("Invalid factor for adapting alpha (>1): {}".format(alpha_factor))
        if conf_level <= 0 or conf_level >= 1:
            raise ValueError("Invalid value for confidence level (0,1): {}".format(conf_level))
        if stopping not in ('never', 'first'):
            raise ValueError("stopping not in ('never', 'first'){}".format(stopping))
        if increase_crit not in ('great', 'great_equal', 'heuristic'):
            raise ValueError("increase_crit not in ('great', 'great_equal', 'heuristic'): {}".format(increase_crit))
        
        self.optimizer = optimizer
        self.state = defaultdict(dict)
    
        # State initialization: leaky bucket belongs to global state.
        p = self.optimizer.param_groups[0]['params'][0]
        self.state['bucket_A'] = LeakyBucket(adapt_period, leak_ratio, p.dtype, p.device) # TODO: what happens with multiple GPUs?? p.device??
        self.state['bucket_C'] = LeakyBucket(adapt_period, leak_ratio, p.dtype, p.device)
        
        # Other things
        self.state['alpha'] = lr
        self.state["tot_steps"] = 0
        self.state["stopping"] = stopping
        self.state["increase_crit"] = increase_crit
        self.state["stop_increase"] = False
        self.state["max_lr"] = max_lr
        
        # statistics parameters
        self.state['period'] = int(adapt_period)
        self.state['conf_level'] = float(conf_level)
        self.state['var_mode'] = var_mode
        self.state['factor_A'] = float(alpha_factor) 
        self.state['adapt_alpha'] = adapt_alpha
        self.state['nsteps'] = 0
    
    def get_lr(self):
        return self.state["alpha"]
    
    def warmup_finished(self):
        return self.state["stop_increase"]
    
    @torch.no_grad()
    def step(self):
        
        self.state["tot_steps"] += 1
        max_lr = self.state["max_lr"]
        
        # TODO: make these floats, use item() instead of cpu(), and handle division by zero, then test with 'cos'
        full_hg = torch.tensor(0.)
        full_g_k_norm_sqr = torch.tensor(0.)
        full_d_k_1_norm_sqr = torch.tensor(0.)

        for group in self.optimizer.param_groups:
            
            # TODO: loose connection between alpha adn lr_k, improve it
            lr_k = group["lr"]
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                
                if p.grad is None:
                    continue
                
                p_state = self.state[p]
                
                if len(p_state) == 0:
                    p_state['p(k-1)'] = p.detach().clone()
                    p_state['lr(k-1)'] = 1.
                
                # Compute descent direction
                d_k_1 = (p_state['p(k-1)'] - p)/lr_k
                # d_k_1 = (p_state['p(k-1)'] - p)/p_state['lr(k-1)']
                
                # Overwrite previous parameter
                p_state['p(k-1)'] = p.detach().clone()
                p_state['lr(k-1)'] = lr_k
                
                # Grad
                grad = p.grad.detach().clone()
                if weight_decay > 0:
                    grad.add_(p, alpha=weight_decay)
                
                # Hypergrad
                full_hg += torch.dot(grad.view(-1), d_k_1.view(-1)).cpu()
                
                # ||g||^2 and ||d||^2
                full_g_k_norm_sqr += torch.dot(grad.view(-1), grad.view(-1)).cpu()
                full_d_k_1_norm_sqr += torch.dot(d_k_1.view(-1), d_k_1.view(-1)).cpu()
        
        # Normalized hypergradient
        full_cos = full_hg / (full_g_k_norm_sqr.sqrt() * full_d_k_1_norm_sqr.sqrt())
        
        # Add statistics to leaky bucket
        if self.state["tot_steps"] > 1:
            self.state['bucket_A'].add(full_hg)
            self.state['bucket_C'].add(full_cos)
        
        # log signals on wandb
        if wandb.run is not None:
            wandb.log({
                'StatsHGIncreaseLR-hg': full_hg,
                'StatsHGIncreaseLR-cos': full_cos,
                'StatsHGIncreaseLR-g': full_g_k_norm_sqr,
                'StatsHGIncreaseLR-d': full_d_k_1_norm_sqr,
            }, step=self.state["tot_steps"])
        
        # conduct statistical test 
        self.state['nsteps'] += 1
        if self.state['nsteps'] == self.state['period']:
            
            # TMP
            rank_0 = True
            if torch.distributed.is_available():
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() != 0:
                        rank_0 = False
            if rank_0:
                print(f"==> Stat Test at step {self.state['tot_steps']}")

            # Mean, Std
            meanA, stdA, _ = self.state['bucket_A'].mean_std(mode=self.state['var_mode'])
            meanC, stdC, _ = self.state['bucket_C'].mean_std(mode=self.state['var_mode'])
            n_samples = self.state['bucket_A'].count
            
            # Test Statistic
            T_stat_A = meanA/stdA * math.sqrt(n_samples) if stdA!=0. else float("Nan")
            T_stat_C = meanC/stdC * math.sqrt(n_samples) if stdC!=0. else float("Nan")
            
            # Quantile (1-conf_level)/2
            t_val = -stats.t.ppf((1-self.state['conf_level'])/2, n_samples-1)
            
            # Test statistics
            T_stat = None
            mean = None
            if self.state['adapt_alpha'] == 'dot':
                T_stat = T_stat_A
                mean = meanA
            elif self.state['adapt_alpha'] == 'cos':
                T_stat = T_stat_C
                mean = meanC
            
            # TEST
            alpha_changed = False
            if not self.state["stop_increase"] and not math.isnan(T_stat):
                
                # increase while E[hg]>0
                if self.state["increase_crit"] == "great":
                    if T_stat > t_val: # T>t -> E[hg]>0 -> undershoot
                        self.state['alpha'] = min(self.state['alpha'] * self.state['factor_A'], max_lr)
                        alpha_changed = True
            
                # increase while E[hg]>=0
                elif self.state["increase_crit"] == "great_equal":
                    if T_stat > -t_val: # T>-t -> E[hg]>=0 -> undershoot, or optimal
                        self.state['alpha'] = min(self.state['alpha'] * self.state['factor_A'], max_lr)
                        alpha_changed = True

                # increase while E[hg]>0, without statistical test
                elif self.state["increase_crit"] == "heuristic_mean":
                    if mean>0: # means E[hg]>0 -> undershoot
                        self.state['alpha'] = min(self.state['alpha'] * self.state['factor_A'], max_lr)
                        alpha_changed = True
                        
                # stop increasing only if alpha has not changed and stopping=="first", otherwise never stop increasing
                self.state["stop_increase"] = (not alpha_changed) and (self.state["stopping"] == "first")
                
            # Update lr and momentum of each parameter groups
            if alpha_changed:
                for group in self.optimizer.param_groups:
                    group['lr'] = self.state['alpha']

            # Reset buckets and number of steps to zero. Reset momentum buffers? probably not.
            self.state['bucket_A'].reset()
            self.state['bucket_C'].reset()
            self.state['nsteps'] = 0
            
            # TEMP: print
            if alpha_changed:
                rank_0 = True
                if torch.distributed.is_available():
                    if torch.distributed.is_initialized():
                        if torch.distributed.get_rank() != 0:
                            rank_0 = False
                if rank_0:
                    print(f"====> LR changed, {self.state['alpha']}")
            
            # Log test statistics and CI on wandb
            if wandb.run is not None:
                wandb.log({
                    'StatsHGIncreaseLR-T_stat': T_stat, 
                    'StatsHGIncreaseLR-t_quantile': t_val, 
                    'StatsHGIncreaseLR-mean': mean, 
                }, step=self.state["tot_steps"])
                
        # log signals on wandb
        if wandb.run is not None:
            wandb.log({
                'lr-iteration': self.state['alpha'],
                }, step=self.state["tot_steps"])