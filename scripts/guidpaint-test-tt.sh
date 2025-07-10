#!/bin/bash

JUMP_LENGTHS="1 2 5 10" # shell脚本文件使用空格分隔
JUMP_N_SAMPLE="1 2 3 5"
JUMP_START_STEPS="230 229 220 219 210 209 200 199"
JUMP_END_STEPS="0 50 100"

for jump_length in $JUMP_LENGTHS
do
  for jump_n_sample in $JUMP_N_SAMPLE
  do
    for jump_start_step in $JUMP_START_STEPS
    do
      for jump_end_step in $JUMP_END_STEPS
      do
          python two_stage_inpainting.py --config_file configs/config.yaml \
            --input datasets/test \
            --output result \
            --mask datasets/test/masks \
            --labels 11,11 \
            --debug \
            --use_local_guid \
            --stage1.algorithm guidpaint \
            --stage1.use_timetravel \
            --ddim.schedule_params.jump_length $jump_length \
            --ddim.schedule_params.jump_n_sample $jump_n_sample \
            --ddim.schedule_params.jump_start_step $jump_start_step \
            --ddim.schedule_params.jump_end_step $jump_end_step
      done
    done
  done
done