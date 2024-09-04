import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class MODNet(torch.nn.Module):
    # Simplified MODNet class definition
    # (You'll need to implement this based on the MODNet architecture)
    pass

model = MODNet()
model.load_state_dict(torch.load('modnet_photographic_portrait_matting.ckpt', map_location='cpu'))
model.eval()

def prepare_image(image_path):
    im = Image.open(image_path)
    im = im.convert('RGB')
    
    # Resize and normalize the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(im).unsqueeze(0)

def process_image(model, image):
    with torch.no_grad():
        _, _, matte = model(image, True)
    
    return matte[0][0].cpu().numpy()

# Use the functions
image_path = 'image1.jpg'
input_image = prepare_image(image_path)
alpha_matte = process_image(model, input_image)


