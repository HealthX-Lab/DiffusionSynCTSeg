#!/bin/bash
module load python
python -V
#mkdir -p '/home/rtm/scratch/rtm/data/MedImagepreprocess/ventricle_labels/'

neurogliaSubmit -I  /home/rtm/deeplearning_gpu.simg -j Quick python  /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/threshold.py -100 1000