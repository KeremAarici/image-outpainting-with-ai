import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

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
    # Standard VGG normalization multiplier.
    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, x, y):
        # Gram Matrix prevents FP16 overflow in matrix multiplication
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Dynamic Device Detection
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

            x_fp32 = (x.float() - mean) / std
            y_fp32 = (y.float() - mean) / std

            x_h1, y_h1 = self.slice1(x_fp32), self.slice1(y_fp32)
            x_h2, y_h2 = self.slice2(x_h1), self.slice2(y_h1)
            x_h3, y_h3 = self.slice3(x_h2), self.slice3(y_h2)
            x_h4, y_h4 = self.slice4(x_h3), self.slice4(y_h3)

            perc_loss = self.l1(x_h1, y_h1) + self.l1(x_h2, y_h2) + self.l1(x_h3, y_h3) + self.l1(x_h4, y_h4)

            style_loss = (
                self.l1(self.gram_matrix(x_h1), self.gram_matrix(y_h1)) +
                self.l1(self.gram_matrix(x_h2), self.gram_matrix(y_h2)) +
                self.l1(self.gram_matrix(x_h3), self.gram_matrix(y_h3)) +
                self.l1(self.gram_matrix(x_h4), self.gram_matrix(y_h4))
            )

        return perc_loss, style_loss

class MaskBoundaryLoss(nn.Module):
    """Maske birleşim hattındaki dikiş izlerini (seam) ve renk kırılmalarını engelleyen kayıp fonksiyonu."""
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.l1 = nn.L1Loss()
        
        # Gradyan/Kenar sürekliliği için Sobel filtreleri
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, fake_img, target, mask):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            fake_fp32 = fake_img.float()
            target_fp32 = target.float()
            mask_fp32 = mask.float()

            # MaxPool2d (Morfolojik Dilation ve Erosion) ile sınır şeridini çıkarma
            dilated = F.max_pool2d(mask_fp32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
            eroded = -F.max_pool2d(-mask_fp32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
            boundary_strip = dilated - eroded  # Sadece sınır çizgisinde 1, diğer yerlerde 0 olan bant

            # 1. Sınır Hattı Renk Uyumu (Piksel L1)
            loss_pixel = self.l1(fake_fp32 * boundary_strip, target_fp32 * boundary_strip)

            # 2. Sınır Hattı Gradyan/Doğrultu Uyumu (Sobel Filtresi)
            b, c, h, w = fake_fp32.shape
            fake_grad_x = F.conv2d(fake_fp32.view(b * c, 1, h, w), self.sobel_x, padding=1).view(b, c, h, w)
            target_grad_x = F.conv2d(target_fp32.view(b * c, 1, h, w), self.sobel_x, padding=1).view(b, c, h, w)
            
            fake_grad_y = F.conv2d(fake_fp32.view(b * c, 1, h, w), self.sobel_y, padding=1).view(b, c, h, w)
            target_grad_y = F.conv2d(target_fp32.view(b * c, 1, h, w), self.sobel_y, padding=1).view(b, c, h, w)

            loss_grad = (
                self.l1(fake_grad_x * boundary_strip, target_grad_x * boundary_strip) +
                self.l1(fake_grad_y * boundary_strip, target_grad_y * boundary_strip)
            )

            return loss_pixel + 2.0 * loss_grad