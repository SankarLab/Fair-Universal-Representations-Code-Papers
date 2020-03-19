#!/bin/bash

#SBATCH -N 1  #number of nodes
#SBATCH -n 1  #number of tasks
#SBATCH -c 27  #number of cores
#SBATCH --mem=40000  #amt in megabytes of memory required
#SBATCH -t 0-12:20:00  #time in d:hh:mm:ss
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p serial
#SBATCH -q normal
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops,or fails
#SBATCH --mail-user=hladdha@asu.edu # send-to add

source activate tf1.12-gpu
python script_multiprocessing.py
exit