# CONFIG FOR HGWARMUP FULL TRAINING

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = 'out/out-fullrun'

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-fullrun'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 600
log_interval = 10

# weight decay
weight_decay = 1e-1

# hgwarmup
learning_rate = 1e-10
adapt_alpha = "dot"
increase_crit = "great_equal"
conf_level = .9
alpha_factor = 2.
adapt_period = 300
dtype = "float16"

wandb_id = wandb_run_name + "-{}-{}-{}-{}-{}-{}".format(
    learning_rate,
    adapt_alpha,
    increase_crit,
    conf_level,
    alpha_factor,
    adapt_period,
)