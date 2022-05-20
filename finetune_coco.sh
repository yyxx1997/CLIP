export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco.yaml \
    --output_dir output/Retrieval_coco/${model_name} \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/${model_name}.pt \
    --gradient_accumulation_steps 2 \
    --eval_before_train