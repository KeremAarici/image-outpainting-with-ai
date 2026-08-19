import torch
import matplotlib.pyplot as plt
from dataset import OutpaintingDataset
from models import UNetGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/gen_epoch_10.pth"  # You can select the any model

def predict_and_show(index=0):
    # 1. Install dataset and model
    dataset = OutpaintingDataset("data/val2017", image_size=256)
    masked_img, mask, target = dataset[index]

    generator = UNetGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    generator.eval()

    # 2. Add Batch Size (1, C, H, W)
    masked_img_input = masked_img.unsqueeze(0).to(DEVICE)
    mask_input = mask.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = generator(masked_img_input, mask_input)

    # 3. Denormalize Tensors [-1, 1] → [0, 1] and Convert to Matplotlib Format
    masked_img_np = (masked_img.permute(1, 2, 0).cpu().numpy() + 1) / 2
    target_np = (target.permute(1, 2, 0).cpu().numpy() + 1) / 2
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2

    # Print the images side by side 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(masked_img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(output_np)
    axes[1].set_title("AI Output")
    axes[1].axis("off")

    axes[2].imshow(target_np)
    axes[2].set_title("Target Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # (0 to 4999 ) You can try the any image that you like
    predict_and_show(index=12)