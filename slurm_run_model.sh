#!/bin/bash

#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=20         # Number of tasks per node
#SBATCH --job-name=initial_chain     # Job name
#SBATCH --mail-user=oliviarose004@protonmail.com  # Email for notifications
#SBATCH --mail-type=ALL              # Get notifications for all events
#SBATCH --mem-per-cpu=500MB          # Memory per CPU
#SBATCH --time=10:00:00              # Wall time

# Purge existing modules to avoid conflicts
module purge

# Load necessary modules
module load gcc/12.2.0
module load openmpi/4.1.4

# Activate your Python environment
source activate NGC5102

# Run the Python script with MPI
mpirun -n 20 python JAM/model_a.py

