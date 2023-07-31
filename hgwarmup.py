from collections import defaultdict
from scipy import stats
import math
import torch
import torch.nn.functional as F
import wandb


def is_master():
    rank_0 = True
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                rank_0 = False
    return rank_0


class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        """
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all,
               2 - keep most recent 1/2,
               3 - keep most recent 1/3,
               ...
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        """
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0  # number of elements kept in queue (excluding leaked)
        self.start = 0  # count = end - start
        self.end = 0  # start and end are pointer to the memory chunck
        self.total = 0  # total number of elements added (including leaked)

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
        if self.end == self.size:  # when the end index reach size
            self.double_size()  # double the size of buffer

        self.buffer[self.end] = val  # always put new value at the end
        self.end += 1  # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1  # increase count in queue by one
            else:  # otherwise leak and keep same count
                self.start += 1  # increase start index by one

        self.total += 1  # always increase total count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0 : self.count] = self.buffer[self.start : self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode="bm"):
        mean = torch.mean(self.buffer[self.start : self.end]).item()

        if mode == "bm":  # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(
                self.buffer[self.start : self.end].unsqueeze(0).unsqueeze(0),
                kernel_size=b_n,
                stride=b_n,
            ).view(-1)
            centered = Yks - mean
            std = math.sqrt(b_n / (len(Yks) - 1)) * torch.norm(centered).item()
            dof = b_n - 1
        elif mode == "olbm":  # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(
                self.buffer[self.start : self.end].unsqueeze(0).unsqueeze(0),
                kernel_size=b_n,
                stride=1,
            ).view(-1)
            centered = Yks - mean
            std = (
                math.sqrt(b_n * self.count / (len(Yks) * (len(Yks) - 1)))
                * torch.norm(centered).item()
            )
            dof = self.count - b_n
        else:  # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start : self.end]).item()
            dof = self.count - 1

        return mean, std, dof


class HGWarmup(object):
    def __init__(
        self,
        optimizer,
        lr,
        adapt_period=100,
        adapt_alpha="dot",
        alpha_factor=2.0,
        conf_level=0.95,
        max_lr=100.0,
        increase_crit="great",
        leak_ratio=1,
        var_mode="bm",
    ):
        self.optimizer = optimizer
        self.state = defaultdict(dict)

        self.state["lr"] = lr
        self.state["signal"] = adapt_alpha
        self.state["period"] = int(adapt_period)
        self.state["lr_factor"] = float(alpha_factor)
        self.state["max_lr"] = max_lr

        self.state["step"] = 0

        p = self.optimizer.param_groups[0]["params"][0]
        self.state["bucket"] = LeakyBucket(adapt_period, leak_ratio, p.dtype, p.device)
        self.state["conf_level"] = conf_level
        self.state["var_mode"] = var_mode

        self.state["increase_crit"] = increase_crit
        self.state["stop_increase"] = False

    def get_lr(self):
        return self.state["lr"]

    def warmup_finished(self):
        return self.state["stop_increase"]

    @torch.no_grad()
    def compute_signal(self):
        # TODO: make these floats, use item() instead of cpu()
        full_hg = torch.tensor(0.0)

        for group in self.optimizer.param_groups:
            # TODO: loose connection between alpha adn lr_k, improve it
            lr_k = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                p_state = self.state[p]

                if len(p_state) == 0:
                    p_state["p(k-1)"] = p.detach().clone()

                # Compute descent direction
                d_k_1 = (p_state["p(k-1)"] - p) / lr_k

                # Overwrite previous parameter
                p_state["p(k-1)"] = p.detach().clone()

                # Negative Hypergrad
                full_hg += torch.dot(
                    p.grad.detach().clone().view(-1), d_k_1.view(-1)
                ).cpu()

        # Log signal
        if wandb.run is not None:
            wandb.log(
                {
                    "stat/signal": full_hg,
                },
                step=self.state["step"],
            )

        return full_hg

    @torch.no_grad()
    def step(self):
        self.state["step"] += 1

        # store signal in bucket
        signal = self.compute_signal()
        self.state["bucket"].add(signal)

        if self.state["step"] % self.state["period"] == 0:
            # Mean, Std
            mean, std, dof = self.state["bucket"].mean_std(mode=self.state["var_mode"])
            n_samples = self.state["bucket"].count

            # Quantile (1-conf_level) for *unilateral* test
            t_val = -stats.t.ppf((1 - self.state["conf_level"]), dof)

            # Test statistics
            T_stat = mean / std * math.sqrt(n_samples)

            # Log
            if wandb.run is not None and is_master():
                wandb.log(
                    {
                        "stat/T_stat": T_stat,
                        "stat/t_quantile": t_val,
                        "stat/mean": mean,
                        "lr-iteration": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.state["step"],
                )

            # Reset bucket
            self.state["bucket"].reset()

            # Return if we already found optimal LR
            if self.state["stop_increase"]:
                return

            ### Statistical Test ###
            increase_LR = False

            # increase while E[hg]>0
            if self.state["increase_crit"] == "great":
                if T_stat > t_val:
                    increase_LR = True

            # increase while E[hg]>=0
            elif self.state["increase_crit"] == "great_equal":
                if T_stat > -t_val:
                    increase_LR = True

            # increase while E[hg]>0, without statistical test
            elif self.state["increase_crit"] == "heuristic":
                if mean > 0:
                    increase_LR = True

            # always increase (for testing purposes only!)
            elif self.state["increase_crit"] == "always":
                increase_LR = True

            # if no test fired, we keep LR and stop increasing
            if not increase_LR:
                self.state["stop_increase"] = True
                return

            # otherwise, we increase LR
            lr_factor = self.state["lr_factor"]
            max_lr = self.state["max_lr"]
            for group in self.optimizer.param_groups:
                group["lr"] = min(group["lr"] * lr_factor, max_lr)
                self.state["lr"] = min(self.state["lr"] * lr_factor, max_lr)

    def state_dict(self):
        """Returns the state as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the state.

        Args:
            state_dict (dict): state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
