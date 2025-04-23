#!/bin/bash
#SBATCH --job-name=physiokinematic
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --time=00:60:00     # Specify maximum runtime 
#SBATCH --mem=4GB           # Adjust memory requirements if necessary
#SBATCH --ntasks=1          # Single task per job
#SBATCH --cpus-per-task=4   # Number of CPU cores per task (20 cpus per node /5 cpus per task = 4 models per node)
#SBATCH --array=0-947%60  # total number of HII regions (index range) 80% means 20 nodes will be taken up (100%=25)

eval "$(conda shell.bash hook)"
conda activate physiokinematic

idx0=$(( 2*$SLURM_ARRAY_TASK_ID ))
idx1=$(( 2*$SLURM_ARRAY_TASK_ID + 1 ))

tmpdir=`mktemp -d`
echo "starting to analyze $idx0"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python ~/physiokinematic/sample_physiokinematic.py $idx0
mv "trace_${idx0}.pkl" results/.
rm -rf $tmpdir

tmpdir=`mktemp -d`
echo "starting to analyze $idx1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python ~/physiokinematic/sample_physiokinematic.py $idx1
mv "trace_${idx1}.pkl" results/.
rm -rf $tmpdir
