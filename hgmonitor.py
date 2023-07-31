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


class HGWarmup(object):
    def __init__(
        self,
        optimizer,
    ):
        self.optimizer = optimizer
        self.state = defaultdict(dict)
        self.state["step"] = 0

    @torch.no_grad()
    def step(self):
        self.state["step"] += 1

        full_hg = torch.tensor(0.0)
        full_g_norm_sq = torch.tensor(0.0)
        full_d_norm_sq = torch.tensor(0.0)

        for group in self.optimizer.param_groups:
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

                # ||g||^2 and ||d||^2
                full_g_norm_sq += torch.dot(
                    p.grad.detach().clone().view(-1), p.grad.detach().clone().view(-1)
                ).cpu()
                full_d_norm_sq += torch.dot(d_k_1.view(-1), d_k_1.view(-1)).cpu()

        # Cosine = normalized hypergradient
        full_cos = (
            full_hg / (full_g_norm_sq.sqrt() * full_d_norm_sq.sqrt())
            if (full_g_norm_sq.sqrt() * full_d_norm_sq.sqrt())
            else torch.tensor(float("nan"))
        )

        # Log signal
        if wandb.run is not None and is_master():
            wandb.log(
                {
                    "stat/full_hg": full_hg,
                    "stat/full_cos": full_cos,
                    "stat/full_g_norm_sq": full_g_norm_sq,
                    "stat/full_d_norm_sq": full_d_norm_sq,
                },
                step=self.state["step"],
            )

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
