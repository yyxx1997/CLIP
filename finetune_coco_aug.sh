############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=3,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco
subtask=DataAug_v2
aug_output_dir=../Output/CLIP/${task}/${subtask}/aug

python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/${task}_aug.yaml \
    --output_dir $aug_output_dir \
    --checkpoint ../Models/common/${model_name}.pt \
    --gradient_accumulation_steps 2 \
    --mode aug  \
    --logging_strategy no   \
    --aug_model_ckpt $aug_output_dir/augnet.pth

mode=('task' 'union')

# Iterate over hyperparameter combinations
for mm in "${mode[@]}"; do
    echo $mm
    python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
        --config ./configs/${task}.yaml \
        --output_dir ../Output/CLIP/${task}/${subtask}/${mm} \
        --checkpoint ../Models/common/${model_name}.pt \
        --gradient_accumulation_steps 2 \
        --mode ${mm}    \
        --aug_model_ckpt $aug_output_dir/augnet.pth
done