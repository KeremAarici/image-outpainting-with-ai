from dataset import OutpaintingDataset


dataset = OutpaintingDataset("data/val2017", image_size=256)
masked_img, mask, target = dataset[0]

print(f"Total image number: {len(dataset)}")
print(f"Masked image tensor: {masked_img.shape}") # [3, 256, 256]
print(f"Mask tensor: {mask.shape}")                 # [1, 256, 256]
print(f"Target Visual Tensor: {target.shape}")       # [3, 256, 256]