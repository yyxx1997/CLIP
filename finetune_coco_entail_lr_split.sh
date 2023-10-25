export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'

# Define hyperparameters
lowlr=(0.0 0.3 0.5)

for lr in "${lowlr[@]}"; do
    echo $lr
    python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval_entail_lr_split.py \
    --config ./configs/Retrieval_coco_entail_lr.yaml \
    --output_dir output/Retrieval_coco_entail_lr_split_${lr}/${model_name} \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/${model_name}.pt \
    --eval_before_train \
    --lowlr $lr
done

