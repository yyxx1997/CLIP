export CUDA_VISIBLE_DEVICES=3,4,6,7
model_name='ViT-B-32'
python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco.yaml \
    --output_dir ../Output/debug/Retrieval_coco/${model_name} \
    --checkpoint ../Models/common/${model_name}.pt \
    --gradient_accumulation_steps 2