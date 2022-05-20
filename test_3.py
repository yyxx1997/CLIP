import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(clip.available_models())
# checkpoint = torch.load("/data1/yx/suda/image-text/sotas/CLIP/output/common/ViT-B-32.pt", map_location='cpu') 
model, preprocess = clip.load("/data1/yx/suda/image-text/sotas/CLIP/output/common/ViT-B-32.pt", device=device)

output_dir = './output/common'
save_obj = {
        'state_dict': model.state_dict(),
    }
torch.save(save_obj, os.path.join(output_dir, 'test.pt'))

model, preprocess = clip.load(os.path.join(output_dir, 'test.pt'), device=device)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]