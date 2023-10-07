import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = ann['caption']
        extra_info = (ann['image'],caption)
        return image, caption, extra_info
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        filter_text = set()
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            image_name=ann['image']
            if image_name == "unk" or "":
                for i, caption in enumerate(ann['caption']):
                    if caption in filter_text:
                        continue
                    filter_text.add(caption)
                    self.text.append(caption)
                    self.txt2img[txt_id] = -1
                    txt_id += 1
            else:
                self.image.append(image_name)
                self.img2txt[img_id] = []
                for i, caption in enumerate(ann['caption']):
                    if caption in filter_text:
                        continue
                    filter_text.add(caption)
                    self.text.append(caption)
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
class re_random_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = random.sample(self.ann,1)[0]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = ann['caption']
        return image, caption
            
class re_entail_lr_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = ann['caption']
        gold = ann['gold'] 

        return image, caption, gold
    
class re_entail_lr_split_train_dataset(Dataset):
    def __init__(self, ann_file, entailments, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.entailments = json.load(open(entailments,'r'))
        self.img2txt_entail = {}
        self.goldens = {}
        for image_path, dct in self.entailments.items():
            goldens = dct['goldens']
            self.goldens[image_path] = goldens
            entailments = dct['entailments']
            if entailments:
                self.img2txt_entail[image_path] = entailments
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_name = ann['image']
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = ann['caption']
        if image_name in self.img2txt_entail.keys():
            entail_pools = self.img2txt_entail[image_name]
            random_entail = random.sample(entail_pools,1)[0]
        elif image_name in self.goldens.keys():
            random_entail = random.sample(self.goldens[image_name],1)[0]
        else:
            random_entail = caption

        return image, caption, random_entail, image_name

    def random(self):
        ann = random.sample(self.ann,1)[0]
        caption = ann['caption']
        return caption