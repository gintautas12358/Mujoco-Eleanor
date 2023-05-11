#!/bin/bash
#SBATCH --gres=gpu:1

singularity exec --nv singularity/eleanor.sif python3 RL/popsan/sac_cuda_norm.py --env $1 --server True