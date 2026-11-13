#!/bin/bash
#SBATCH --job-name=VDM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80000mb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/Training_gen_%j.log
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gpus=1


module purge
# module restore <your modules>  # EDIT: site-specific (original used an Lmod collection)
module load texlive

source /path/to/your/venv/bin/activate  # EDIT: your Python environment
date

echo "Training on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks, each with $SLURM_CPUS_PER_TASK cores."

#srun python -u train_flows.py
srun python -u generate_data.py

date
