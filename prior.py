import numpy as np
from PIL import Image
from data.celeba_data import read_imgs

imgs = read_imgs("/data/ziz/not-backed-up/jxu/CelebA/celeba32-train")
print(imgs.shape)
