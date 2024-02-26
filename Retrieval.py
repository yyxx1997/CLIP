import argparse
from ast import arg
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import clip
from tqdm import tqdm
from contextlib import nullcontext
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from clip.AugmentNet import AugNet


def train(model, model_without_ddp, train_loader, val_loader, test_loader, train_args):
    
    max_epoch = config.schedular['epochs']
    batch_size_train = config.batch_size_train
    K = config.gradient_accumulation_steps
    logging_steps = config.logging_steps
    logging_strategy = config.logging_strategy
    ckpt_output_path = config.ckpt_output_path
    max_grad_norm = config.max_grad_norm
    metrics = config.metrics
    optimizer, lr_scheduler = train_args['optimizer'], train_args['lr_scheduler']
    start_epoch = train_args.get('start_epoch', 1)
    total_step = train_args.get('total_step', 0)
    total_train_batch_size = batch_size_train * K * config.world_size

    best_scores = defaultdict(lambda:None)
    scaler = GradScaler()

    metric_logger = utils.MetricLogger(logging=logger.info, delimiter=" - ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    logger.info("Start training")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {max_epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_train}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {K}")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch+1):
        model.train()
        logger.info(" -" * 20 + "Epochs of [{}/{}]".format(epoch, max_epoch) + " - " * 20)
        header = 'Train Epoch: [{}/{}]'.format(epoch, max_epoch)
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        for i, (image, text, _) in enumerate(metric_logger.log_every(train_loader, header=header)):
            image_input = image.to(device,non_blocking=True)   
            text_input = clip.tokenize(text,truncate=True).to(device)  
            
            sync_context = model.no_sync if config.local_rank != -1 and (i + 1) % K != 0 else nullcontext
            amp_context = autocast if config.use_amp else nullcontext
            with sync_context():
                with amp_context():
                    loss = model(image_input,text_input)               
                    loss = loss / K
                scaler.scale(loss).backward()

            if (i + 1) % K == 0:
                # Best practice of AMP in DDP framework:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#functions-that-need-a-particular-dtype
                # https://zhuanlan.zhihu.com/p/165152789
                scaler.unscale_(optimizer)                
                # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
                # https://blog.csdn.net/zhaohongfei_358/article/details/122820992
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()

            total_step += 1
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item()*K)
            metric_logger.synchronize_between_processes()
            need_tb_logs = metric_logger.latest_meter(prefix='train/')
            
            if (logging_strategy == "epoch" and i == len(train_loader) - 1) or (logging_strategy == "steps" and total_step % logging_steps == 0):
                val_stats, val_prediction = evaluate(model_without_ddp, val_loader)
                test_stats, test_prediction = evaluate(model_without_ddp, test_loader, "Test") if not config.only_dev else (val_stats, val_prediction)

                save_evidence = []
                for metric_name in metrics:
                    if metric_name not in val_stats.keys():
                        continue
                    score = best_scores[metric_name]
                    current_score = float(val_stats[metric_name])
                    if score is None or current_score > score:
                        save_evidence.append(metric_name)
                        best_scores[metric_name] = current_score

                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'step': i+1,
                             'total_step': total_step
                             }

                need_tb_logs.update({
                    **{f'val/{k}': v for k, v in val_stats.items()},
                    **{f'test/{k}': v for k, v in test_stats.items()}
                })

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'step': i+1,
                    'total_step': total_step
                }

                if utils.is_main_process():
                    ckpt_sub_path = os.path.join(ckpt_output_path, f"epoch_{epoch}-step_{i}")

                    # logging statements
                    utils.write_json(ckpt_sub_path, "log_stats", log_stats)

                    # logging prediction
                    utils.write_json(ckpt_sub_path, "val_prediction", val_prediction)
                    utils.write_json(ckpt_sub_path, "test_prediction", test_prediction)

                    # Saving normal checkpoints
                    if save_evidence or config.save_every_checkpoint:
                        torch.save(save_obj, os.path.join(ckpt_sub_path, 'checkpoint.pth'))

                    # Saving checkpoints if they are distinct
                    for metric_name in save_evidence:
                        best_ckpt_path = os.path.join(ckpt_output_path, f"best_{metric_name}")
                        utils.copy_whole_dir(ckpt_sub_path, best_ckpt_path)

            tb_writer.add_dict_scalar(need_tb_logs, total_step)
            lr_scheduler.step(total_step)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()
        torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger.summary(mode="avg")))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('***** Stopping training *****')
    logger.info('Training time {}'.format(total_time_str))
    tb_writer.close()


@torch.no_grad()
def evaluate(model, data_loader, special_name="Val"):
    logger.info("- - - - - - - - - - - - - Evaluation- - - - - - - - - - - - - ")
    # test
    model.eval()
    metric_logger = utils.MetricLogger(logging=logger.info, delimiter=" - ")
    header = f'Evaluating {special_name} Set: '

    texts = data_loader.dataset.text
    images = data_loader.dataset.image
    img2txt = data_loader.dataset.img2txt
    txt2img = data_loader.dataset.txt2img
    image_features = []
    text_features= []
    image_ids = []

    num_text = len(texts)
    text_bs = config['batch_size_test']
    for i in metric_logger.log_every(range(0, num_text, text_bs), header=header + "Loading text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = clip.tokenize(text,truncate=True).to(device)
        text_output = model.encode_text(text_input)
        text_features.append(text_output.to('cpu'))

    text_features = torch.cat(text_features, dim=0)

    for image, img_id in metric_logger.log_every(data_loader, header=header + "Loading image features..."):
        image_input = image.to(device)
        image_output = model.encode_image(image_input)
        image_features.append(image_output.to('cpu'))
        image_ids.extend(img_id.tolist())

    image_features = torch.cat(image_features, dim=0)

    # normalized features
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logger.info("- - - - - - - - - - - - - Calculating Results- - - - - - - - - - - - - ")
    result = {}
    topk_upper = config['k_test']
    scores_i2t = logits_per_image.cpu().numpy()
    scores_t2i = logits_per_text.cpu().numpy()

    # Get topk retrieval results
    for img in range(scores_i2t.shape[0]):
        contents = []
        for wait_check in np.argsort(scores_i2t[img])[::-1][:topk_upper]:
            contents.append(int(wait_check))
        result[image_ids[img]] = contents

    topk_result = {}
    for image_id, txt_ids in result.items():
        topk_result[images[image_id]] = {
            "goldens": [texts[txtid] for txtid in img2txt[image_id]], 
            "topks": [texts[txtid] for txtid in txt_ids]
            }
    # Images->Text
    pres = np.zeros((scores_i2t.shape[0], 10))
    ranks = np.zeros(scores_i2t.shape[0])
    golden_total=0
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = img2txt[index]
        golden_total+=len(goldens)
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        pres[index] = np.cumsum(np.in1d(inds[:10], goldens))

    # Compute metrics

    pr5 = 100.0 * np.sum(pres[:, 4]) / golden_total
    pr10 = 100.0 * np.sum(pres[:, 9]) / golden_total

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        copes = np.where(inds == txt2img[index])[0]
        if len(copes) == 0:
            ranks[index] = 0
        else:
            ranks[index] = copes[0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    pr_mean = (pr5 + pr10)/2
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'i2t_r1': tr1,
                   'i2t_r5': tr5,
                   'i2t_r10': tr10,
                   'i2t_r_mean': tr_mean,
                   'i2t_pr5': pr5,
                   'i2t_pr10': pr10,
                   'i2t_pr_mean': pr_mean,
                   't2i_r1': ir1,
                   't2i_r5': ir5,
                   't2i_r10': ir10,
                   't2i_r_mean': ir_mean,
                   'r_mean': r_mean
                   }
    logger.info(eval_result)
    return eval_result, topk_result

def main():

    #### Model ####
    train_args = {}
    logger.info("- - - - - - - - - - - - - Creating model- - - - - - - - - - - - - ")
    clip_model, preprocess = clip.load(config.checkpoint, device="cpu", download_root=config.download_root)
    clip_model = clip_model.float()
    
    if config.mode == "re":
        logger.info("No loading requirement.")
        model = clip_model
    else:
        model = AugNet(clip_model, 6, mode=config.mode)
        if config.mode == 'aug':
            checkpoint = torch.load(config.task_model_ckpt)
            msg = model.clip.load_state_dict(checkpoint['model'])
        elif config.mode == 'union' or config.mode == 'task':
            checkpoint = torch.load(config.aug_model_ckpt)
            msg = model.semantic_encoder.load_state_dict(checkpoint['model'])
        logger.info(msg)
        
    model = model.to(device)   
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    #### Dataset ####
    logger.info("- - - - - - - - - - - - - Creating dataset- - - - - - - - - - - - - ")
    train_dataset, val_dataset, test_dataset = create_dataset('re', preprocess, config)  

    if config.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
    # next(iter(train_loader))

    #### Training Controler ####
    logger.info("- - - - - - - - - - - - - Loading TrainArgs- - - - - - - - - - - - - ")
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche.steps_per_epoch = len(train_loader)
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    train_args['optimizer'] = optimizer
    train_args['lr_scheduler'] = lr_scheduler

    if config.eval_before_train:
        logger.info("- - - - - - - - - - - - - Evaluate Before Train- - - - - - - - - - - - - ")
        evaluate(model_without_ddp, val_loader)
        evaluate(model_without_ddp, test_loader, special_name="Test")

    train(model, model_without_ddp, train_loader, val_loader,
                  test_loader, train_args)    
    if config.mode == "aug":
        save_obj = {
            'model': model_without_ddp.semantic_encoder.state_dict(),
        }

        if utils.is_main_process():
            torch.save(save_obj, os.path.join("../Models", f'{config.mode}.pth'))

    logger.info("- - - - - - - - - - - - - End of All- - - - - - - - - - - - - ")         


def parse_args():
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )   
    parser.add_argument('--aug_model_ckpt', default="../Models/augnet.pth")   
    parser.add_argument('--task_model_ckpt', default="../Output/CLIP/Retrieval_coco/DataAug/re/casnmt/2024-02-26-13-26/checkpoints/best_r_mean/checkpoint.pth") 
    parser.add_argument('--mode', type=str, choices=['aug','task','union', 're'], default='task')
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='./output/Retrieval_coco_debug')        
    parser.add_argument('--checkpoint', default="ViT-B/32")   
    parser.add_argument('--download_root', default="./output/common") 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--only_dev', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--max_grad_norm', default=5.0, type=float,
                        help='clip gradient norm of an iterable of parameters')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='device number of current process.')
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--logging_strategy', type=str, choices=['no','epoch','steps'], default='epoch')
    parser.add_argument('--logging_level', type=str, choices=['DEBUG','INFO','ERROR','WARNING'], default='INFO')
    parser.add_argument('--save_every_checkpoint', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()
    return args
            
if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = utils.read_yaml(args.config)
    config = utils.AttrDict(config)
    args = utils.AttrDict(args.__dict__)
    # The parameters passed in from the command line take precedence
    config.update(args)

    # Determine global parameters and settings
    utils.init_distributed_mode(config)
    device = torch.device(config.device)
    # fix the seed for reproducibility
    utils.setup_seed(config.seed)
    # record them in file.
    current_branch, git_info = utils.get_git_info(os.path.dirname(os.path.abspath(__file__)))
    config.current_branch = current_branch
    logger, tb_writer = utils.create_logger(config)

    logger.info(f"Here is all global configuration:\n {str(config)}")
    logger.info(f"Here is all git repo infomation:\n {git_info}")

    main()
