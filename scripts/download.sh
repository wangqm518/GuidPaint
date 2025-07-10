#!/bin/bash

(
# download pretrained models
mkdir -p checkpoints
cd checkpoints

# model pretrained on ImageNet
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt  # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt

# model pretrained on CelebA
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX

# model pretrained on Places
gdown https://drive.google.com/uc?id=1QEl-btGbzQz6IwkXiFGd49uQNTUtTHsk
)

prepare_celeba(){
    BASENAME="lama-celeba"
    mkdir -p $BASENAME
    # 重命名压缩包
    mv celebahq-resized-256x256.zip celeba_hq_256.zip
    # 解压缩数据集
    unzip celeba_hq_256.zip -d ${BASENAME}


    # 重命名，从0开始，10# 为10进制
    for i in `echo {00000..29999}`
    do
        mv ${BASENAME}'/celeba_hq_256/'$i'.jpg' ${BASENAME}'/celeba_hq_256/'$[10#$i]'.jpg'
    done

    # Split: split train -> train & val，拆分数据集
    cat lama_split/train_shuffled.flist | shuf > ${BASENAME}/temp_train_shuffled.flist
    cat ${BASENAME}/temp_train_shuffled.flist | head -n 2000 > ${BASENAME}/val_shuffled.flist
    cat ${BASENAME}/temp_train_shuffled.flist | tail -n +2001 > ${BASENAME}/train_shuffled.flist
    cat lama_split/val_shuffled.flist > ${BASENAME}/visual_test_shuffled.flist

    mkdir ${BASENAME}/train_256/
    mkdir ${BASENAME}/val_source_256/
    mkdir ${BASENAME}/visual_test_source_256/
    # 移动文件
    cat ${BASENAME}/train_shuffled.flist | xargs -I {} mv ${BASENAME}/celeba_hq_256/{} ${BASENAME}/train_256/
    cat ${BASENAME}/val_shuffled.flist | xargs -I {} mv ${BASENAME}/celeba_hq_256/{} ${BASENAME}/val_source_256/
    cat ${BASENAME}/visual_test_shuffled.flist | xargs -I {} mv ${BASENAME}/celeba_hq_256/{} ${BASENAME}/visual_test_source_256/
}

# download datasets
(
mkdir -p datasets
cd datasets
# celebahq data
gdown https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256
sleep 1
prepare_celeba
rm celeba_hq_256.zip

# imagenet data
# you can find it in datasets/imagenet100
)