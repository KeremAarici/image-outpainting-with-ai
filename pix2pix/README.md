The goal of this project is to provide artificial intelligence with a photograph, first help it understand the image, then have it assess the overall context, and finally enable it to fill in the missing parts of the photograph—whether inward or outward—and to teach it how to do so.

First i started this project with Pix2pix which known as a structure to do works like i mentioned at the previous paragraph. But i belive that i just reached the limits of this structure because actually pix2pix structure was not designed to understand the general context it just fills the blank part with correct colors it never understands what is the exact object which causes of incapability of generating an object with no clue. That’s why, for example, if there’s a fence in the part of the image you provided that isn’t visible, it can never recognize it as a fence. As a result, it never draws a fence there in the final output; it simply shifts the colors of the preceding fence to that area, which leaves us with nothing but blurry colors.

As a result of all this i decided to change the structure from pix2pix to latent diffusion which is more successfull at generating images from nothing.

This is an example of pix2pix structure
<img width="1034" height="776" alt="The Limit Of Pix2pix Structure" src="https://github.com/user-attachments/assets/cd6e8f28-a1b3-43d0-9e38-53df98eede53" />

## Datasets Used

This project utilizes the following open-source datasets for training, testing, and fine-tuning:

* **[COCO (Common Objects in Context)](https://cocodataset.org/)**: Used `train2017` and `val2017` splits for image outpainting, inpainting, and complex scene generation tasks.
* **[CMP Facade Database](https://cmp.felk.cvut.cz/~stylewa/facades/)**: Used during the initial Pix2Pix baseline setup for architectural segmentation and image-to-image translation experiments.

