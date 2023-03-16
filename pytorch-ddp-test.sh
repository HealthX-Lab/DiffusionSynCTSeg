#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out
#SBATCH --cpus-per-task=4

module load python/3.10 # Using Default Python version - Make sure to choose a version that suits your application
module load cuda
#source ENV_monai/bin/activate
source /home/rtm/projects/rrg-eugenium/rtm/domain_adaptation_CTscan/ENV_monai/bin/activate
nvidia-smi
export CUDA_LAUNCH_BLOCKING=1
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
#export CUDA_LAUNCH_BLOCKING=1
srun python main.py