#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=csr-cpu-spmv

srun build/SpMV ../data/identity.mtx
