#!/bin/bash

#SBATCH --job-name=nz32r02h5
#SBATCH --account=torch_pr_513_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mp6191@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --comment="gpu_mps=yes"

# Purge modules to be safe
module purge

# Activate singularity and run script
juliaGPU driver.jl
