#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=csr-gpu-spmv
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

module load CUDA/12.1
srun build/SpMV ../data/mawi_201512020330.mtx
