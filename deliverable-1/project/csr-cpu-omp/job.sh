#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=csr-cpu-spmv
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

MAT_FILE=$1
srun build/SpMV $MAT_FILE
