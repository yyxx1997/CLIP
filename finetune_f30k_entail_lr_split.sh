export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name='ViT-B-32'

# Define hyperparameters
lowlr=(0.0 0.3 0.5)

for lr in "${lowlr[@]}"; do
    echo $lr
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 29505 --use_env Retrieval_entail_lr_split.py \
    --config ./configs/Retrieval_f30k_entail_lr.yaml \
    --output_dir output/Retrieval_f30k_entail_lr_split_${lr}/${model_name} \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/${model_name}.pt \
    --eval_before_train \
    --lowlr $lr
done

