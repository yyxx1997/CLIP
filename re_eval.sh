#!/bin/bash
# 'RN50' 'RN101' 'RN50x4' 'RN50x16' 'RN50x64' 'ViT-B/32' 'ViT-B/16' 'ViT-L/14' 'ViT-L/14@336px'
# L="RN50
# RN101
# RN50x4
# RN50x16
# RN50x64
# ViT-B/32
# ViT-B/16
# ViT-L/14
# ViT-L/14@336px"

counter=4 

for i in 'RN50' 'RN101' 'RN50x4' 'RN50x16' 'RN50x64' 'ViT-B/32' 'ViT-B/16' 'ViT-L/14' 'ViT-L/14@336px'
do
echo $i,$counter
CUDA_VISIBLE_DEVICES=$counter python Retrieval.py \
    --config ./configs/Retrieval_f30k.yaml \
    --output_dir output/Retrieval_f30k_${i} \
    --checkpoint $i \
    --evaluate & \
counter=$(((counter+1)%7))
done