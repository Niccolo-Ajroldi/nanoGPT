import submitit

ngpu = 8
timeout = 2 * 60

proj = "nanoGPT"
name_job = "hg-prova-1"

job_folder = "/private/home/niccoloajroldi/runs/stadia/" + proj + "/" + name_job

executor = submitit.AutoExecutor(folder=job_folder)
executor.update_parameters(
    name = name_job,
    tasks_per_node = 1,
    timeout_min = timeout, 
    slurm_partition = "devlab",
    # slurm_partition = "learnlab",
    mem_gb = 512,
    slurm_cpus_per_task = 10 * ngpu,
    slurm_gpus_per_task = ngpu, 
)

# ------------------------------------------------------------------
# SUBMIT
# ------------------------------------------------------------------
        
cmnd = [
    "python", "train.py", "config/train_gpt2.py",
    "--adapt_period=20",
    "--compile=False"
]

function = submitit.helpers.CommandFunction(cmnd)
job = executor.submit(function)
