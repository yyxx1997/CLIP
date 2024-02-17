############################### MixGen ###############################

export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco_mixgen
mix_rate=(0.1 0.2 0.3 0.4 0.5)
mix_lam=(0.5 0.3 0.7)

# Iterate over hyperparameter combinations
for mr in "${mix_rate[@]}"; do
    for ml in "${mix_lam[@]}"; do
        echo $mr, $ml
        python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
            --config ./configs/${task}.yaml \
            --output_dir ../Output/CLIP/${task}/${model_name}/${mr}_${ml} \
            --checkpoint ../Models/common/${model_name}.pt \
            --gradient_accumulation_steps 1 \
            --mix_rate ${mr}    \
            --mix_lam ${ml}
    done
done
