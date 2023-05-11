#!/bin/bash
#SBATCH --gres=gpu:1
srun singularity exec --nv cuda-11.4.2-cudnn8-devel-ubuntu20.04_lava.sif python3 eleanor_train.py

