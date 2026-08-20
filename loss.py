import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        # We move the FFT calculation outside of AMP (autocast) and set it to float32 precision
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.float()
            y_fp32 = y.float()

            x_fft = torch.fft.rfft2(x_fp32, norm="ortho")
            y_fft = torch.fft.rfft2(y_fp32, norm="ortho")

            x_mag = torch.abs(x_fft)
            y_mag = torch.abs(y_fft)

            return self.l1(x_mag, y_mag)
class PerceptualAndStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)]) # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)]) # relu3_3
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # relu4_3
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()

    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, x, y):
        x_h1, y_h1 = self.slice1(x), self.slice1(y)
        x_h2, y_h2 = self.slice2(x_h1), self.slice2(y_h1)
        x_h3, y_h3 = self.slice3(x_h2), self.slice3(y_h2)
        x_h4, y_h4 = self.slice4(x_h3), self.slice4(y_h3)

        # Perceptual loss
        perc_loss = self.l1(x_h1, y_h1) + self.l1(x_h2, y_h2) + self.l1(x_h3, y_h3) + self.l1(x_h4, y_h4)

        # Style loss (Gram Matrix - Texture Details)
        style_loss = (
            self.l1(self.gram_matrix(x_h1), self.gram_matrix(y_h1)) +
            self.l1(self.gram_matrix(x_h2), self.gram_matrix(y_h2)) +
            self.l1(self.gram_matrix(x_h3), self.gram_matrix(y_h3)) +
            self.l1(self.gram_matrix(x_h4), self.gram_matrix(y_h4))
        )

        return perc_loss, style_loss