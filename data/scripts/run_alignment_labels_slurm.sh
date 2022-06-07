#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH --cpus-per-task 8         # Cpus per task requested
#SBATCH --ntasks 2        #
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb

python3 -u ../alignment.py > alignment_slurm.log

