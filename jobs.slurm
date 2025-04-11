#!/bin/bash
#SBATCH --job-name=physiokinematic
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --time=00:10:00     # Specify maximum runtime 
#SBATCH --mem=4GB          # Adjust memory requirements if necessary
#SBATCH --ntasks=1          # Single task per job
#SBATCH --cpus-per-task=4   # Number of CPU cores per task 
#SBATCH --array=0-1894      # total number of HII regions (index range)

# Load required modules
module load python/3.8  

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate physiokinematic

# Run the Python script for the specific HII region index
python ~/physiokinematic/sample_physiokinematic.py $SLURM_ARRAY_TASK_ID