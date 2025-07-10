#!/bin/bash

MASK_TYPE="half expand square" # shell脚本文件使用空格分隔
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

      cmd="python two_stage_inpainting.py --config_file configs/config.yaml \
        --input datasets/${dataset} \
        --output results/${mask_type}/guidpaint \
        --mask_type ${mask_type} \
        --stage1.config_path configs/${conf_type}.yaml \
        --stage1.algorithm guidpaint "

      # 根据条件添加额外参数
      if [ "$conf_type" == "imagenet" ]; then
          cmd="$cmd --use_pred_y --pred_y_top_k 1 --stage1.use_guidance"
      fi

      # 执行命令
      eval $cmd
  done
done