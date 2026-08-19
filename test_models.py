import torch
from models import UNetGenerator, PatchGANDiscriminator


device = "cuda" if torch.cuda.is_available() else "cpu"


dummy_masked_img = torch.randn(2, 3, 256, 256).to(device)
dummy_mask = torch.ones(2, 1, 256, 256).to(device)
dummy_target_img = torch.randn(2, 3, 256, 256).to(device)


generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)


fake_img = generator(dummy_masked_img, dummy_mask)
print(f"Generator Successful! Output Size: {fake_img.shape}") 
# Expected: [2, 3, 256, 256]


disc_out = discriminator(dummy_masked_img, dummy_mask, fake_img)
print(f"Discriminator Successful! Output Size: {disc_out.shape}") 
# Expected: [2, 1, 30, 30]