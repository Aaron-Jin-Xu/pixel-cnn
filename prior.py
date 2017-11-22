import numpy as np
from PIL import Image
from data.celeba_data import read_imgs

imgs = read_imgs("/data/ziz/not-backed-up/jxu/CelebA/celeba32-train")
imgs = np.array([imgs==i for i in range(256)], dtype=np.uint32)
imgs = imgs.sum(axis=1)
np.savez_compressed("prior", arr=imgs)

print(np.load("prior.npz")["arr"].shape)
