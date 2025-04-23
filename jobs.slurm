#!/bin/bash
#SBATCH --job-name=physiokinematic
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --time=00:30:00     # Specify maximum runtime 
#SBATCH --mem=4GB          # Adjust memory requirements if necessary
#SBATCH --ntasks=1          # Single task per job
#SBATCH --cpus-per-task=5   # Number of CPU cores per task (20 cpus per node /5 cpus per task = 4 models per node)
#SBATCH --array=0-1894%80      # total number of HII regions (index range) 80% means 20 nodes will be taken up (100%=25)

eval "$(conda shell.bash hook)"
conda activate physiokinematic

tmpdir=`mktemp -d`
echo "starting to analyze $SLURM_ARRAY_TASK_ID"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python ~/physiokinematic/sample_physiokinematic.py $SLURM_ARRAY_TASK_ID
rm -rf $tmpdir

