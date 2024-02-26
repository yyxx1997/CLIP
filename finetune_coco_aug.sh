############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco
subtask=DataAug

python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/${task}_aug.yaml \
    --output_dir ../Output/CLIP/${task}/${subtask}/aug \
    --checkpoint ../Models/common/${model_name}.pt \
    --gradient_accumulation_steps 1 \
    --mode aug  \
    --logging_strategy no   \
    --task_model_ckpt ../Output/CLIP/Retrieval_coco/DataAug/re/casnmt/2024-02-26-13-26/checkpoints/best_r_mean/checkpoint.pth

mode=('task' 'union')

# Iterate over hyperparameter combinations
for mm in "${mode[@]}"; do
    echo $mm
    python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
        --config ./configs/${task}.yaml \
        --output_dir ../Output/CLIP/${task}/${subtask}/${mm} \
        --checkpoint ../Models/common/${model_name}.pt \
        --gradient_accumulation_steps 1 \
        --mode ${mm}    \
        --aug_model_ckpt ../Models/aug.pth
done