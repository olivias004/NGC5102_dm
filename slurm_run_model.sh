#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name=initial_chain
#SBATCH --mail-user=oliviarose004@protonmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=10:00:00
#SBATCH --account=oz059

module purge
module load gcc/12.2.0
module load openmpi/4.1.4
source activate NGC5102

# Run script and redirect output to logs
mpirun -n 20 python JAM/model_a.py > output.log 2> error.log
