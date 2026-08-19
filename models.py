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
        # 256x256 -> 128x128
        self.e1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1) 
        # 128x128 -> 64x64
        self.e2 = UNetBlock(64, 128, down=True)       
        # 64x64 -> 32x32
        self.e3 = UNetBlock(128, 256, down=True)      
        # 32x32 -> 16x16
        self.e4 = UNetBlock(256, 512, down=True)      
        # 16x16 -> 8x8 (Darboğaz / Bottleneck)
        self.e5 = UNetBlock(512, 512, down=True)      

        # --- DECODER ---
        # 8x8 -> 16x16
        self.d1 = UNetBlock(512, 512, down=False, use_dropout=True) 
        # 16x16 -> 32x32 (Girdi kanalı Skip Connection yüzünden 512+512=1024 olur)
        self.d2 = UNetBlock(1024, 256, down=False)                  
        # 32x32 -> 64x64 (256+256 = 512)
        self.d3 = UNetBlock(512, 128, down=False)                   
        # 64x64 -> 128x128 (128+128 = 256)
        self.d4 = UNetBlock(256, 64, down=False)                    

        # 128x128 -> 256x256 (Son Katman)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Compress the pixels in between [-1, 1]
        )

    def forward(self, masked_img, mask):
        
        x = torch.cat([masked_img, mask], dim=1) 
        
        # Encoder Flow
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # Decoder Flow and Skip Connections
        d1 = self.d1(e5)
        d2 = self.d2(torch.cat([d1, e4], dim=1)) # e4 with d1 
        d3 = self.d3(torch.cat([d2, e3], dim=1)) # e3 with d2 
        d4 = self.d4(torch.cat([d3, e2], dim=1)) # e2 with d3 

        return self.final(torch.cat([d4, e1], dim=1)) # e1 with d4 


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
            *block(in_channels, 64, stride=2, normalize=False), 
            *block(64, 128, stride=2),
            *block(128, 256, stride=2),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, padding=1) 
        )

    def forward(self, masked_img, mask, image):
        
        x = torch.cat([masked_img, mask, image], dim=1)
        return self.model(x)