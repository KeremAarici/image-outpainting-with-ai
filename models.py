import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """
    Base Layer that unet uses.
    It can make encoding and decoding
    """
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            # Encoder Layer
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            # Decoder Layer
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UNetGenerator(nn.Module):
    """
    Generator
    Input : Masked iamge (3) + Mask (1) = Total 4 channel
    Output : Completed RGB Image = Total 3 channel
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        
        # --- ENCODER ---
        def down_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.e1 = down_block(in_channels, 64, normalize=False) # 256 -> 128
        self.e2 = down_block(64, 128)                           # 128 -> 64
        self.e3 = down_block(128, 256)                          # 64 -> 32
        self.e4 = down_block(256, 512)                          # 32 -> 16
        self.e5 = down_block(512, 512)                          # 16 -> 8
        self.e6 = down_block(512, 512)                          # 8 -> 4
        self.e7 = down_block(512, 512)                          # 4 -> 2
        self.bottleneck = down_block(512, 512, normalize=False) # 2 -> 1      

        # --- DECODER ---
        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.d1 = up_block(512, 512, dropout=True)  # 1 -> 2
        self.d2 = up_block(1024, 512, dropout=True) # 2 -> 4
        self.d3 = up_block(1024, 512, dropout=True) # 4 -> 8
        self.d4 = up_block(1024, 512)               # 8 -> 16
        self.d5 = up_block(1024, 256)               # 16 -> 32
        self.d6 = up_block(512, 128)                # 32 -> 64
        self.d7 = up_block(256, 64)                 # 64 -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Çıktıyı [-1, 1] aralığına sıkıştırır
        )

    def forward(self, masked_img, mask):
        
        x = torch.cat([masked_img, mask], dim=1) 
        
        # Encoder Flow
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b = self.bottleneck(e7)

        # Decoder Flow and Skip Connections
        d1 = torch.cat([self.d1(b), e7], dim=1)
        d2 = torch.cat([self.d2(d1), e6], dim=1)
        d3 = torch.cat([self.d3(d2), e5], dim=1)
        d4 = torch.cat([self.d4(d3), e4], dim=1)
        d5 = torch.cat([self.d5(d4), e3], dim=1)
        d6 = torch.cat([self.d6(d5), e2], dim=1)
        d7 = torch.cat([self.d7(d6), e1], dim=1) 

        return self.final(d7)


class PatchGANDiscriminator(nn.Module):
    """
    Discriminator
    Input : Masked Image (3) + Mask (1) + Image (3) = 7 Channels
    Output : An N x N patch score map (Reality Map)
    """
    def __init__(self, in_channels=7):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, stride=2, normalize=False), # 256 -> 128
            *block(64, 128, stride=2),                          # 128 -> 64
            *block(128, 256, stride=2),                         # 64 -> 32
            *block(256, 512, stride=1),                         # 32 -> 31
            nn.Conv2d(512, 1, kernel_size=4, padding=1)         # 31 -> 30
        )

    def forward(self, masked_img, mask, image):
        
        x = torch.cat([masked_img, mask, image], dim=1)
        return self.model(x)