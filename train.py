import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OutpaintingDataset
from models import UNetGenerator, PatchGANDiscriminator
from loss import PerceptualLoss

# 1-Hyperparameters and settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8          # RTX 5060 (8GB VRAM) Ideal batch sie
LEARNING_RATE = 0.0002   # Pix2Pix original learning rate
LAMBDA_L1 = 100         # L1 loss mass multiplier
LAMBDA_PERCEPTUAL = 10
NUM_EPOCHS = 25         # Total loop number
IMAGE_SIZE = 256
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_fn():
    # 2-Data Loader
    dataset = OutpaintingDataset("data/val2017", image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # 3-Starting the models
    gen = UNetGenerator().to(DEVICE)
    disc = PatchGANDiscriminator().to(DEVICE)

    # 4-Optimization Algorithms
    # Pix2Pix suggests beta1=0.5, beta2=0.999 
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # 5=Loss func.
    BCE = nn.BCEWithLogitsLoss() # GAN real/fake loss (Sigmoid included)
    L1_LOSS = nn.L1Loss()        # Pixel-based loss of detail
    PERCEPTUAL_LOSS = PerceptualLoss().to(DEVICE)

    print(f"Training starts in {DEVICE.upper()}... Total Pictures: {len(dataset)}")

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for idx, (masked_img, mask, target) in enumerate(loop):
            masked_img = masked_img.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            # ---------------------
            #  1. Discriminator
            # ---------------------
            fake_img = gen(masked_img, mask)

            disc_real = disc(masked_img, mask, target)
            loss_disc_real = BCE(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(masked_img, mask, fake_img.detach())
            loss_disc_fake = BCE(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # ---------------------
            #  2. Generator
            # ---------------------
            disc_fake_for_gen = disc(masked_img, mask, fake_img)
            loss_gen_gan = BCE(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))

            # Masked L1 Loss: Üretilen alana (mask == 0) 10 kat daha fazla odaklanılır
            missing_region_mask = 1.0 - mask
            l1_global = L1_LOSS(fake_img, target)
            l1_masked = L1_LOSS(fake_img * missing_region_mask, target * missing_region_mask)
            loss_gen_l1 = (l1_global + 10 * l1_masked) * LAMBDA_L1

            # VGG19 Perceptual Loss
            loss_gen_perc = PERCEPTUAL_LOSS(fake_img, target) * LAMBDA_PERCEPTUAL

            # Total Generator Loss
            loss_gen = loss_gen_gan + loss_gen_l1 + loss_gen_perc

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(
                D_loss=loss_disc.item(),
                G_loss=loss_gen.item(),
                L1=loss_gen_l1.item(),
                Perc=loss_gen_perc.item()
            )

        # Save the model weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(gen.state_dict(), f"{CHECKPOINT_DIR}/gen_epoch_{epoch+1}.pth")
            torch.save(disc.state_dict(), f"{CHECKPOINT_DIR}/disc_epoch_{epoch+1}.pth")
            print(f"\n[Saved weights] Epoch {epoch+1} checkpoint created.")

if __name__ == "__main__":
    train_fn()