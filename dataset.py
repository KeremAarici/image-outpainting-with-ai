import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class OutpaintingDataset(Dataset):
    def __init__(self, image_dir="data/val2017", image_size=256):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_size = image_size
        
        # appropriate for the exit of Pix2Pix Generator Tanh  [-1, 1] normalization
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.transform(image)  # Original picture (Ground Truth)
        
        # Mask: 1.0 = Secured place, 0.0 = Area to Be Expanded
        mask = torch.ones((1, self.image_size, self.image_size))
        
        # Random corner sellection (0: right, 1: left, 2: top, 3: bottom)
        edge = random.randint(0, 3)
        mask_ratio = random.uniform(0.2, 0.4)
        pixels = int(self.image_size * mask_ratio)

        if edge == 0:   
            mask[:, :, -pixels:] = 0
        elif edge == 1: 
            mask[:, :, :pixels] = 0
        elif edge == 2: 
            mask[:, :pixels, :] = 0
        elif edge == 3: 
            mask[:, -pixels:, :] = 0

    
        masked_image = target * mask

        return masked_image, mask, target