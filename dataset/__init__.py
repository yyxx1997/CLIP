import torch
from torch.utils.data import DataLoader

from dataset.caption_dataset import *


def create_dataset(dataset, preprocess, config): 
    
    if dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], preprocess, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], preprocess, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], preprocess, config['image_root'])                
        return train_dataset, val_dataset, test_dataset  
    elif dataset=='re_entail':
        train_dataset = re_train_dataset(config['train_file'], preprocess, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], preprocess, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], preprocess, config['image_root'])    
        random_texts = re_random_dataset(config['train_file'], preprocess, config['image_root'])            
        return train_dataset, val_dataset, test_dataset,random_texts


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    