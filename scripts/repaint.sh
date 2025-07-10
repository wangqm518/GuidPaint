#!/bin/bash

MASK_TYPE="expand half square" # shell脚本文件使用空格分隔
DATASET="imagenet100 celebahq_test_256_decoration"

for dataset in $DATASET
do
  for mask_type in $MASK_TYPE
  do
      if [ "$dataset" == "imagenet100" ]; then
          conf_type="imagenet"
      elif [ "$dataset" == "celebahq_test_256_decoration" ]; then
          conf_type="celebahq"
      fi

      python two_stage_inpainting.py --config_file configs/config.yaml \
        --input datasets/${dataset} \
        --output results/${mask_type}/repaint \
        --mask_type $mask_type \
        --stage1.config_path configs/${conf_type}.yaml \
        --stage1.algorithm repaint
  done
done