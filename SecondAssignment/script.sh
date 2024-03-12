#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single node  
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# Distribute processes in round-robin fashion
#SBATCH --distribution=cyclic
#
# Run the job on one node, all cores on the same node (full node)
#SBATCH --ntasks=4 --nodes=1
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:30:00
#
# Job name
#SBATCH --job-name=assignment1
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

#Load a module which provides mpi
module load anaconda/python-3.10.9/2023.03
module load mpi/2021.6.0
#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program
mpirun -np $SLURM_NTASKS python3 my_code.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
