#!/bin/bash
#SBATCH -A naiss2026-4-1108-gpu
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -o /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_%x_%j.out
#SBATCH -e /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet/results/slurm_%x_%j.err
# usage: sbatch -J <tag> scripts/run_ft.sh <swiss46|scratch> <seed> [--freeze-bn]
module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate bwtreenet
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
echo "START $SLURM_JOB_NAME $(date)"
python scripts/25_train_finetune.py --init "$1" --seed "$2" --tag "$SLURM_JOB_NAME" --degrade --batch 8 $3
echo "END $SLURM_JOB_NAME $(date)"
