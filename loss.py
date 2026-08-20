import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGG19 based Perceptual Loss
    It compares the haigh level texture detail of the output.
    """
    def __init__(self):
        super().__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # We take the first 16 layers of the VGG (high-level features up to the rel4_2 layer)
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)]).eval()
        
        for param in self.slice.parameters():
            param.requires_grad = False  # We stop th VGG weights

        # ImageNet Normalizasyon constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # It is mapped from the range [-1, 1] to the range [0, 1] and normalized using the ImageNet normalization scheme
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        feat_x = self.slice(x)
        feat_y = self.slice(y)

        return nn.functional.l1_loss(feat_x, feat_y)