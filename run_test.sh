#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=32G
#SBATCH --time=0-04:00
##SBATCH --output=%N-%j.out

module load python/3.10 # Using Default Python version - Make sure to choose a version that suits your application
source ./domain_adaptation_CTscan/ENV_monai2/bin/activate # path to python environment
nvidia-smi
#sbatch --account=rrg-eugenium pytorch-ddp-test.sh

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
export CUDA_LAUNCH_BLOCKING=1
export CUDA_ERROR_CHECK=1


srun python sb_test.py # sb_test.py |cut_test.py