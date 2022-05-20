export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_f30k.yaml \
    --output_dir output/Retrieval_f30k/ViT-L-14-336px \
    --checkpoint /data1/yx/suda/image-text/sotas/CLIP/output/common/ViT-L-14-336px.pt \
    --gradient_accumulation_steps 2 \
    --eval_before_train