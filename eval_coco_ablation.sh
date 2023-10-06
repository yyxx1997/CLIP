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
    python Retrieval.py \
        --config ./configs/Retrieval_eval_coco.yaml \
        --output_dir /data1/yx/suda/image-text/sotas/CLIP/output/Retrieval_coco_rebuild_entail_lr_split_${sh_list[i]}/ViT-B-32 \
        --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/Retrieval_coco_rebuild_entail_lr_split_${sh_list[i]}/ViT-B-32/checkpoint_best.pth \
        --evaluate
done;
