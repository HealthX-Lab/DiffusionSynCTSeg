#!/bin/bash
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/iDB
target_ct_path=( $( ls -a ${BASE_PATH}/*ct.nii.gz))

for i in $(seq ${#target_ct_path[@]}); do
    target_ct=${target_ct_path[i-1]}
    echo $target_ct
    module load python
    python -V
    neurogliaSubmit -I  /home/rtm/deeplearning_gpu.simg -j Quick python  /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/changetype.py $target_ct
done