export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_name='ViT-B-32'
python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco_rebuild_entail.yaml \
    --output_dir output/Retrieval_coco_rebuild_entail/${model_name} \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/${model_name}.pt \
    --eval_before_train