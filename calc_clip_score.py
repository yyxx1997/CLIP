import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm

from pathlib import Path
from torch.utils.data import DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
device = torch.device('cuda')
import clip
import heapq    


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
        for ann in tqdm(self.ann, total=len(self.ann), desc="Loading dataset..."):
            image_name=ann['image']
            if image_name not in self.image:
                img_id = len(self.image)
                self.image.append(image_name)
                self.img2txt[img_id] = []
            else:
                img_id = self.image.index(image_name)
            caption = ann['caption']
            if caption not in self.text:
                self.txt2img[cnt] = []
                txt_id = cnt
                self.text.append(caption)
                cnt += 1
            else:
                txt_id = self.text.index(caption)
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id].append(img_id)  
        assert len(self.image) == len(self.img2txt)
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index


@torch.no_grad()
def evaluate(model, data_loader, output_path):
    print("- - - - - - - - - - - - - Evaluation- - - - - - - - - - - - - ")
    # test
    model.eval()

    texts = data_loader.dataset.text
    images = data_loader.dataset.image

    text_features_all= []

    num_text = len(texts)
    text_bs = 2048
    for i in tqdm(range(0, num_text, text_bs), total=len(range(0, num_text, text_bs)), desc="Loading text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = clip.tokenize(text,truncate=True).to(device)
        text_output = model.encode_text(text_input)
        text_features_all.append(text_output.to('cpu'))

    text_features_all = torch.cat(text_features_all, dim=0)
    logit_scale = model.logit_scale.exp()
    topk_upper = 128

    for image, img_ids in tqdm(data_loader, total=len(data_loader), desc="Loading image features & Calc i2t sim per batch..."):
        image_input = image.to(device)
        image_features = model.encode_image(image_input)
        image_ids = img_ids.tolist()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        image_heaps = {}

        for i in range(0, num_text, text_bs):
            text_features = text_features_all[i: min(num_text, i+text_bs)]
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = text_features.to(device)
        
            # cosine similarity as logits
            
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            scores_i2t = logits_per_image.cpu().numpy()
            scores_i2t = np.float16(scores_i2t).tolist()

            for img_id, score in zip(image_ids, scores_i2t):
                if img_id not in image_heaps:
                    image_heaps[img_id] = []
                img_hp = image_heaps[img_id]
                img_hp.extend(list(zip(score, range(i, min(num_text, i+text_bs)))))
                image_heaps[img_id] = heapq.nlargest(topk_upper, img_hp)

        results = []
        # Get topk retrieval results
        for img_id, topk_scores in image_heaps.items():
            image_name = images[img_id]
            topk_result = {}
            
            for score, txt_id in topk_scores:
                topk_result[texts[txt_id]] = score

            body = {
                "image": image_name,
                "topk_scores": topk_result
            }
            results.append(body)
        
        with open(output_path, 'a+') as file:
            for dct in results:
                file.write(json.dumps(dct)+'\n')



if __name__ == "__main__":
    trainset_path = "../Data/re_data/coco_train.json"
    image_root = "../Data/images"

    model, preprocess = clip.load("../Models/common/ViT-B-32.pt", device=device)
    ds = re_eval_dataset(trainset_path, preprocess, image_root)
    data_loader = DataLoader(
        ds,
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    ) 

    
    model = model.float()
    model = model.to(device)
    checkpoint = torch.load("../Output/CLIP/Retrieval_coco/DataAug/re/mixgen/2024-02-22-18-59/checkpoints/best_r_mean/checkpoint.pth")
    msg = model.load_state_dict(checkpoint['model'])


    evaluate(model, data_loader, "../Output/CLIP/ts_clip_score.jsonl")