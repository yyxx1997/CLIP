import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, text

class re_train_dataset_mixgen(Dataset):
    def __init__(self, ann_file, transform, image_root, mix_rate=0.25, mix_lam=0.5):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.mix_rate = mix_rate
        self.mix_lam = mix_lam
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = ann['caption']
        return image, caption
    
    def collate_fn(self, batchs):
        images = []
        texts = []
        for image, text in batchs:
            images.append(image)
            texts.append(text)
        images = torch.stack(images)
        N = int(self.mix_rate * len(batchs))
        mixgen(images, texts, N, self.mix_lam)
        return images, texts

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = ann['caption']
        return image, caption


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        cnt = 0 
        for img_id, ann in enumerate(self.ann):
            image_name=ann['image']
            self.image.append(image_name)
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                if caption not in self.text:
                    self.txt2img[cnt] = []
                    txt_id = cnt
                    self.text.append(caption)
                    cnt += 1
                else:
                    txt_id = self.text.index(caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id].append(img_id)  
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
