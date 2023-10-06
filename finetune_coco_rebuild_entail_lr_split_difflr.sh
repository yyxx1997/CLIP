#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_name='ViT-B-32'


sh_list=(
  0.1
  0.2
  0.4
  0.5
  0.6
  0.7
  0.8
  0.9
  1.0
)
 
for(( i=0;i<${#sh_list[@]};i++)) 
 
do
    echo ${sh_list[i]};
    python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval_entail_lr_split.py \
    --config ./configs/Retrieval_coco_rebuild_entail_lr.yaml \
    --output_dir output/Retrieval_coco_rebuild_entail_lr_split_${sh_list[i]}/${model_name} \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/${model_name}.pt \
    --eval_before_train \
    --lowlr ${sh_list[i]}
done;