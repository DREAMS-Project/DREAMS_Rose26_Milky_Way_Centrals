#!/bin/bash
#SBATCH --job-name=TNG50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem=950gb
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/log_%j.log
#SBATCH --constraint=rome
#SBATCH -p cmbas


module restore mpi
source /path/to/your/venv/bin/activate  # EDIT: your Python environment
mpirun -n 15 python -u disk_size.py
