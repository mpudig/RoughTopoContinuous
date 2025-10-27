#!/bin/bash

#SBATCH --job-name=<JOBNAME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mp6191@nyu.edu
#SBATCH --output=slurm_%j.out

# Purge modules to be safe
module purge

# Activate singularity and run script
julia fresh_nc_driver.jl