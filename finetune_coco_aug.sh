############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco
subtask=DataAug_v3
mode=union

alpha=(1 0.5 0.1 0.01)
temp=(1 0.7 0.5)

# Iterate over hyperparameter combinations
for al in "${alpha[@]}"; do
    for tp in "${temp[@]}"; do
        combination=al${al}_tp${tp}
        echo $combination
        python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
            --config ./configs/${task}.yaml \
            --output_dir ../Output/CLIP/${task}/${subtask}/${mode}/$combination \
            --checkpoint ../Models/common/${model_name}.pt \
            --gradient_accumulation_steps 2 \
            --mode ${mode}  \
            --alpha $al  \
            --temp  $tp
    done
done

K=(20 10 5 1)
eta=(1 0.6 0.3)

# Iterate over hyperparameter combinations
for mgrc_K in "${K[@]}"; do
    for mgrc_eta in "${eta[@]}"; do
        combination=K${mgrc_K}_eta${mgrc_eta}
        echo $combination
        python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
            --config ./configs/${task}.yaml \
            --output_dir ../Output/CLIP/${task}/${subtask}/${mode}/$combination \
            --checkpoint ../Models/common/${model_name}.pt \
            --gradient_accumulation_steps 2 \
            --mode ${mode}  \
            --mgrc_K $mgrc_K  \
            --mgrc_eta  $mgrc_eta
    done
done