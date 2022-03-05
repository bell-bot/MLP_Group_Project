#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH --cpus-per-task 4	  # Cpus per task requested
#SBATCH --ntasks 2	  # 
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb
#SBATCH --time=0-08:00:00

python3 -u ../link_keywords_threading.py > keywords_slurm.log
