import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image
from tqdm import tqdm


from dataset import OutpaintingDataset
from models import UNetGenerator, PatchGANDiscriminator
from loss import PerceptualAndStyleLoss, FFTLoss, MaskBoundaryLoss, ColorLoss, StructuralGradientLoss

torch.backends.cudnn.benchmark = True


# 1-Hyperparameters and settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8          # RTX 5060 (8GB VRAM) Ideal batch sie
LEARNING_RATE = 0.0002   # Pix2Pix original learning rate
LAMBDA_L1 = 10         # L1 loss mass multiplier
LAMBDA_PERCEPTUAL = 10
LAMBDA_STYLE = 100       # New Texture Loss Weight
LAMBDA_FFT = 100         # New Frequency/Sharpness Weight
LAMBDA_BOUNDARY = 20.0   # Penalty Weight Of Boundary
LAMBDA_COLOR = 15.0
LAMBDA_STRUCT = 20.0
NUM_EPOCHS = 20          # Total loop number
IMAGE_SIZE = 256
CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)


def save_samples(gen, loader, epoch):
    """Eğitim devam ederken örnek görselleri diske kaydeder."""
    gen.eval()
    with torch.no_grad():
        masked_img, mask, target = next(iter(loader))
        masked_img, mask, target = masked_img.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
        # For Fast Conculusion
        with torch.amp.autocast(device_type="cuda"):
            fake_img = gen(masked_img, mask)
        
        # Denormalize [-1, 1] -> [0, 1]
        masked_img = (masked_img + 1) / 2
        fake_img = (fake_img + 1) / 2
        target = (target + 1) / 2

        # İlk 4 görseli grid şeklinde birleştirip kaydet
        results = torch.cat([masked_img[:4], fake_img[:4], target[:4]], dim=0)
        save_image(results, f"{SAMPLES_DIR}/epoch_{epoch+1}.png", nrow=4)
    gen.train()


def train_fn():
    # 2-Data Loader

    dataset_paths = [
    "data/val2017",
    "data/train2017"  # Yeni indirilen klasör
    ]
    SAMPLES_PER_EPOCH = 10000
    sampler = RandomSampler(dataset, replacement=True, num_samples=SAMPLES_PER_EPOCH)

    dataset = OutpaintingDataset(image_dir=dataset_paths, image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,    # RTX 5060 için 8 olarak ayarlamıştık[cite: 3]
        sampler=sampler,          # shuffle=True yerine sampler kullanıyoruz
        num_workers=6,            # İşlemci çekirdeğine göre 6 veya 8 kalabilir
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=3         # GPU'nun veri beklemesini engeller
    )

    # 3-Starting the models
    gen = UNetGenerator().to(DEVICE)
    disc = PatchGANDiscriminator().to(DEVICE)

    # 4-Optimization Algorithms
    # Pix2Pix suggests beta1=0.5, beta2=0.999 
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # AMP Accelerator
    scaler_gen = torch.amp.GradScaler()
    scaler_disc = torch.amp.GradScaler()


    # 5=Loss func.
    BCE = nn.BCEWithLogitsLoss() # GAN real/fake loss (Sigmoid included)
    L1_LOSS = nn.L1Loss()        # Pixel-based loss of detail
    PERCEPTUAL_STYLE_LOSS = PerceptualAndStyleLoss().to(DEVICE)
    FFT_LOSS = FFTLoss().to(DEVICE)
    MASK_BOUNDARY_LOSS = MaskBoundaryLoss(kernel_size=9).to(DEVICE)
    COLOR_LOSS = ColorLoss().to(DEVICE)
    STRUCTURAL_LOSS = StructuralGradientLoss().to(DEVICE)

    # =========================================================
    # To continue your training where you left off!!!!!!!!!!!!!
    # =========================================================
    LOAD_MODEL = True
    START_EPOCH = 50
    ADDITIONAL_EPOCHS = 10

    if LOAD_MODEL:
        print("Loading the available checkpoint weights (Epoch 50)...")
        gen.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/gen_epoch_50.pth"), strict=False)
        disc.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/disc_epoch_50.pth"), strict=False)


    print(f"Training starts in {DEVICE.upper()}... Total Pictures: {len(dataset)}")

    # gen = torch.compile(gen)
    # disc = torch.compile(disc)


    for epoch in range(START_EPOCH, START_EPOCH + ADDITIONAL_EPOCHS):
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{START_EPOCH + ADDITIONAL_EPOCHS}]")

        for idx, (masked_img, mask, target) in enumerate(loop):
            masked_img = masked_img.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            # ---------------------
            #  1. Discriminator
            # ---------------------
            opt_disc.zero_grad(set_to_none=True)

            # Autocast 1 
            with torch.amp.autocast(device_type="cuda"):
                fake_img = gen(masked_img, mask)

                disc_real = disc(masked_img, mask, target)
                real_labels = torch.ones_like(disc_real) * 0.9
                loss_disc_real = BCE(disc_real, real_labels)

                disc_fake = disc(masked_img, mask, fake_img.detach())
                fake_labels = torch.zeros_like(disc_fake)
                loss_disc_fake = BCE(disc_fake, fake_labels)

                loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # Backward pass and optimization scaler
            scaler_disc.scale(loss_disc).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            # ---------------------
            #  2. Generator
            # ---------------------
            opt_gen.zero_grad(set_to_none=True)
            # Autocast 2
            with torch.amp.autocast(device_type="cuda"):
                # Discriminator output for generator
                disc_fake_for_gen = disc(masked_img, mask, fake_img)
                loss_gen_gan = BCE(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))

                # L1 Loss
                missing_region_mask = 1.0 - mask
                l1_global = L1_LOSS(fake_img, target)
                l1_masked = L1_LOSS(fake_img * missing_region_mask, target * missing_region_mask)
                loss_gen_l1 = (l1_global + 10 * l1_masked) * LAMBDA_L1

                # Perceptual and Style Loss
                loss_gen_perc, loss_gen_style = PERCEPTUAL_STYLE_LOSS(fake_img, target)
                loss_gen_perc = loss_gen_perc * LAMBDA_PERCEPTUAL
                loss_gen_style = loss_gen_style * LAMBDA_STYLE

                # Frequency (FFT) Loss. Prevents blurring and smearing
                loss_gen_fft = FFT_LOSS(fake_img, target) * LAMBDA_FFT


                # Boundary
                loss_gen_boundary = MASK_BOUNDARY_LOSS(fake_img, target, mask) * LAMBDA_BOUNDARY
                # Color
                loss_gen_color = COLOR_LOSS(fake_img, target) * LAMBDA_COLOR
                # Geometrical texture and secure
                loss_gen_struct = STRUCTURAL_LOSS(fake_img, target, mask) * LAMBDA_STRUCT

                # Total Generator Loss
                loss_gen = (
                    loss_gen_gan + 
                    loss_gen_l1 + 
                    loss_gen_perc + 
                    loss_gen_style + 
                    loss_gen_fft + 
                    loss_gen_boundary +
                    loss_gen_color +
                    loss_gen_struct
                )

            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            loop.set_postfix(
                D_loss=f"{loss_disc.item():.3f}",
                G_loss=f"{loss_gen.item():.1f}",
                L1=f"{loss_gen_l1.item():.1f}",
                Style=f"{loss_gen_style.item():.1f}",
                Clr=f"{loss_gen_color.item():.1f}",
                Str=f"{loss_gen_struct.item():.1f}"
            )


        if (epoch + 1) % 2 == 0:
            save_samples(gen, loader, epoch)


        # Save the model weights every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(gen.state_dict(), f"{CHECKPOINT_DIR}/gen_epoch_{epoch+1}.pth")
            torch.save(disc.state_dict(), f"{CHECKPOINT_DIR}/disc_epoch_{epoch+1}.pth")
            print(f"\n[Saved weights] Epoch {epoch+1} checkpoint created.")

if __name__ == "__main__":
    train_fn()