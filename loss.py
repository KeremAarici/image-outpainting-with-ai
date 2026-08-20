import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

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
        # Özellik Çıkarımı
        x_h1 = self.slice1(x); y_h1 = self.slice1(y)
        x_h2 = self.slice2(x_h1); y_h2 = self.slice2(y_h1)
        x_h3 = self.slice3(x_h2); y_h3 = self.slice3(y_h2)
        x_h4 = self.slice4(x_h3); y_h4 = self.slice4(y_h3)

        # Perceptual Loss
        perc_loss = self.l1(x_h1, y_h1) + self.l1(x_h2, y_h2) + self.l1(x_h3, y_h3) + self.l1(x_h4, y_h4)

        # Style Loss (Gram Matrix) -> Bulanıklığı yok eden kısım
        style_loss = (
            self.l1(self.gram_matrix(x_h1), self.gram_matrix(y_h1)) +
            self.l1(self.gram_matrix(x_h2), self.gram_matrix(y_h2)) +
            self.l1(self.gram_matrix(x_h3), self.gram_matrix(y_h3)) +
            self.l1(self.gram_matrix(x_h4), self.gram_matrix(y_h4))
        )

        return perc_loss, style_loss