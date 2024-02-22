############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco
sub_task=DataAug

python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/${task}.yaml \
    --output_dir ../Output/CLIP/${task}/${sub_task}/re \
    --checkpoint ../Models/common/${model_name}.pt \
    --gradient_accumulation_steps 1 \
    --eval_before_train \
    --mode re

mix_mode=('mixgen' 'mixgen_batch' 'mixgen_random')
mix_rate=(0.05 0.15 0.25)
mix_lam=(0.5)

# Iterate over hyperparameter combinations
for mm in "${mix_mode[@]}"; do
    for mr in "${mix_rate[@]}"; do
        for ml in "${mix_lam[@]}"; do
            echo $mr, $ml, $mm
            python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
                --config ./configs/${task}.yaml \
                --output_dir ../Output/CLIP/${task}/${sub_task}/${mm}/${mr}_${ml} \
                --checkpoint ../Models/common/${model_name}.pt \
                --gradient_accumulation_steps 1 \
                --eval_before_train \
                --mix_rate ${mr}    \
                --mix_lam ${ml} \
                --mode ${mm}
        done
    done
done