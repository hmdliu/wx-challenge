#!/bin/bash
#SBATCH --job-name=MultiModal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# checkpoint info
exp_id=$1
ckpt_name=$2

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --bind /scratch \
--overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/wx-challenge
python inference.py \
    --ckpt_file save/${exp_id}/${ckpt_name}.bin \
    --test_output_csv pred/${exp_id}_${ckpt_name}.csv
"