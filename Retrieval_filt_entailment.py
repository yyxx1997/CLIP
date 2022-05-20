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
from dataset.utils import pre_caption
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import clip
from tqdm import tqdm
from contextlib import nullcontext
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# def contrastive_loss(logits, dim):
#     neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
#     #import pdb; pdb.set_trace()
#     return -neg_ce.mean()
    
# def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#     #import pdb; pdb.set_trace()
#     image_loss = contrastive_loss(similarity, dim=1)
#     caption_loss = contrastive_loss(similarity, dim=0)
#     return (image_loss + caption_loss) / 2.0

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

def entail_check(images, texts, raw_entailments, max_words=512):
    assert len(images) == len(texts)
    result = []
    i = j = 0
    for i in range(len(images)):
        image = images[i]
        if image not in raw_entailments.keys():
            continue
        entails = raw_entailments[image]
        goldens = [pre_caption(t, max_words) for t in entails['goldens']]
        entailments = [pre_caption(t, max_words)
                       for t in entails['entailments']]
        for j in range(len(texts)):
            if i==j : continue
            text = texts[j]
            text = pre_caption(text, max_words)
            if text in goldens+entailments:
                result.append((i, j))
    return result


def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, random_texts, raw_entailments):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('conflict', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    K = config.gradient_accumulation_steps
    my_context = model.no_sync if config.local_rank != - \
        1 and i % K != 0 else nullcontext
    total = entail = 0
    optimizer.zero_grad()
    for i, (image, text, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        checked = entail_check(extra_info[0], extra_info[1], raw_entailments)
        if checked:
            entail += len(checked)
            for image_id, text_id in checked:
                image_, text_ = random_texts.__getitem__(1)
                image[image_id] = image_
                text[text_id] = text_
        total += len(text)
        image_input = image.to(device,non_blocking=True)
        text_input = clip.tokenize(text,truncate=True).to(device)

        with my_context():
            logits_per_image, _ = model(image_input,text_input)
            loss_itm = clip_loss(logits_per_image)
            loss_itm = loss_itm / K
            loss_itm.backward()  # 积累梯度，不应用梯度改变
        if (i+1) % K == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss_itm=loss_itm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(conflict=entail/total)
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i//step_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


"""
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
"""
@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    print('Computing features for evaluation...')
    print_freq = 50

    texts = data_loader.dataset.text
    images = data_loader.dataset.image
    img2txt = data_loader.dataset.img2txt
    txt2img = data_loader.dataset.txt2img
    image_features = []
    text_features= []
    image_ids = []

    num_text = len(texts)
    text_bs = config['batch_size_test']
    for i in metric_logger.log_every(range(0, num_text, text_bs), print_freq, header + "Loading text and get text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = clip.tokenize(text,truncate=True).to(device)
        text_output = model.encode_text(text_input)
        text_features.append(text_output.to('cpu'))

    text_features = torch.cat(text_features, dim=0)

    for image, img_id in metric_logger.log_every(data_loader, print_freq, header + "Loading image and get image features..."):
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

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'txt_pr5': pr5,
                   'txt_pr10': pr10,
                   'txt_pr_mean': pr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean
                   }
    print("eval_result is:\n",eval_result)
    return eval_result, topk_result

def main(config):
     
    device = torch.device(config.device)
    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #### Model #### 
    print("Creating model")  
    model, preprocess = clip.load(config.checkpoint, device=device, download_root=config.download_root)
    model = model.float()
    model = model.to(device)   
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module  

    #### Dataset #### 
    print("Creating dataset")
    train_dataset, val_dataset, test_dataset, random_texts = create_dataset('re_entail', preprocess, config)  
    raw_entailments=json.load(open(config['entail_file'],'r'))
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
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    # optimizer = optim.Adam(model.parameters(), lr=1e-7,weight_decay=0.2)
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_standard = 'r_mean'

    if config.eval_before_train or config.evaluate:
        val_result,val_topk = evaluation(model_without_ddp, val_loader, device, config)
        test_result,test_topk = evaluation(model_without_ddp, test_loader, device, config)
        if config.evaluate:
            if utils.is_main_process():             
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                            **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(config) + "\n") 
                    f.write(json.dumps(log_stats) + "\n")     
            return
    
    print("Start training")
    print("***** Running training *****")
    print(f"  Num examples = {len(train_loader.dataset)}")
    print(f"  Num Epochs = {max_epoch}")
    print(f"  Instantaneous batch size per device = {config.batch_size_train}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.total_train_batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config, random_texts, raw_entailments)  
            
        val_result,val_topk = evaluation(model_without_ddp, val_loader, device, config)
        test_result,test_topk = evaluation(model_without_ddp, test_loader, device, config)
    
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_result.items()},
                            **{f'test_{k}': v for k, v in test_result.items()},                  
                            'epoch': epoch,
                        }
            with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(config) + "\n") 
                f.write(json.dumps(log_stats) + "\n")   
            
            save_obj = {
                    'state_dict': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            torch.save(save_obj, os.path.join(config.output_dir, 'checkpoint-{}.pth'.format(epoch))) 

            if val_result[best_standard]>best:
                best = val_result[best_standard]    
                best_epoch = epoch
                torch.save(save_obj, os.path.join(config.output_dir, 'checkpoint_best.pth'))
                with open(os.path.join(args.output_dir, "best_top{}_result.json".format(config['k_test'])), "w") as f:
                    f.write(json.dumps(test_topk,ensure_ascii=False,indent=4))

        lr_scheduler.step(epoch+warmup_steps+1)  
        if utils.is_dist_avail_and_initialized():
            dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(config.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

def parse_args():
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )     
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='./output/Retrieval_coco_debug')        
    parser.add_argument('--checkpoint', default="ViT-B/32")   
    parser.add_argument('--download_root', default="./output/common") 
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--local_rank', default=-1, type=int, help='device number of current process.')
    parser.add_argument('--eval_before_train', action='store_true')
    args = parser.parse_args()
    return args
            
if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    utils.init_distributed_mode(args)
    args.total_train_batch_size = config['batch_size_train'] * args.gradient_accumulation_steps * args.world_size
    
    config=utils.AttrDict(config)
    args=utils.AttrDict(args.__dict__)
    config.update(args)

    print("all global configuration is here:\n",config)
    if utils.is_main_process():
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)   
        yaml.dump(dict(config), open(os.path.join(config.output_dir, 'global_config.yaml'), 'w')) 
    main(config)
