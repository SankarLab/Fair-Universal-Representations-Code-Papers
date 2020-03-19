#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 1                        # number of CPU cores to reserve on this compute node

#SBATCH -p asinghargpu1             # Use physicsgpu1 partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue

#SBATCH --gres=gpu:1                # Request two GPUs

#SBATCH -t 3-00:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=hladdha@asu.edu # send-to add

source activate tf1.12-gpu
python code.py
exit