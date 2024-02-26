############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name='ViT-B-32'
task=Retrieval_coco
sub_task=DataAug
mode=mixgen_clip_score

mix_rate=(0.1 0.25 0.3 0.5)
mix_lam=(0.5 0.3 0.7)


# Iterate over hyperparameter combinations
for mr in "${mix_rate[@]}"; do
    for ml in "${mix_lam[@]}"; do
        echo $mr, $ml, $mode
        python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
            --config ./configs/${task}_cscore.yaml \
            --output_dir ../Output/CLIP/${task}/${sub_task}/${mode}/${mr}_${ml} \
            --checkpoint ../Models/common/${model_name}.pt \
            --gradient_accumulation_steps 1 \
            --eval_before_train \
            --mix_rate ${mr}    \
            --mix_lam ${ml} \
            --mode ${mode}
    done
done
