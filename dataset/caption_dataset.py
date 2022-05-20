import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_caption


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
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            image_name=ann['image']
            if image_name == "unk" or "":
                for i, caption in enumerate(ann['caption']):
                    self.text.append(caption)
                    self.txt2img[txt_id] = -1
                    txt_id += 1
            else:
                self.image.append(image_name)
                self.img2txt[img_id] = []
                for i, caption in enumerate(ann['caption']):
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
            

    
