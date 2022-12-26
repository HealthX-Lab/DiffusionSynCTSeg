#!/bin/bash
module load python
python -V


neurogliaSubmit -I  /home/rtm/deeplearning_gpu.simg -j Quick python  /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/extractlabels.py