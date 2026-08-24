import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class OutpaintingDataset(Dataset):
    def __init__(self, image_dir="data/val2017", image_size=256):
        
        if isinstance(image_dir, (list, tuple)):
            self.image_paths = []
            for d in image_dir:
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
                    self.image_paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
        else:
            self.image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
                self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

        self.image_size = image_size
        
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _generate_mask(self, h, w):
        
        mask = np.ones((h, w), dtype=np.uint8)
        
       
        if random.random() < 0.4:
            edge = random.randint(0, 3)
            mask_ratio = random.uniform(0.2, 0.4)
            pixels = int(h * mask_ratio)

            if edge == 0:   
                mask[:, -pixels:] = 0
            elif edge == 1: 
                mask[:, :pixels] = 0
            elif edge == 2: 
                mask[:pixels, :] = 0
            elif edge == 3: 
                mask[-pixels:, :] = 0
        else:
            
            num_strokes = random.randint(3, 8)
            for _ in range(num_strokes):
                pt1 = (int(random.randint(0, w)), int(random.randint(0, h)))
                pt2 = (int(random.randint(0, w)), int(random.randint(0, h)))
                thickness = int(random.randint(12, 35))
                cv2.line(mask, pt1, pt2, color=0, thickness=thickness)
                
            
            if random.random() > 0.5:
                x1 = int(random.randint(0, w // 2))
                y1 = int(random.randint(0, h // 2))
                x2 = int(x1 + random.randint(w // 4, w // 2))
                y2 = int(y1 + random.randint(h // 4, h // 2))
                cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

        return torch.from_numpy(mask).unsqueeze(0).float()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.transform(image)  
        
        _, h, w = target.shape
        mask = self._generate_mask(h, w)
        
        masked_image = target * mask

        return masked_image, mask, target