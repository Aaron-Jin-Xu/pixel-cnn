import numpy as np
from PIL import Image
from data.celeba_data import read_imgs

imgs = read_imgs("/data/ziz/not-backed-up/jxu/CelebA/celeba32-train")
imgs = np.array([(imgs==i).astype(np.uint32) for i in range(256)])
imgs = imgs.sum(axis=1)
np.savez_compressed("/data/ziz/jxu/prior", arr=imgs)

print(np.load("/data/ziz/jxu/prior.npz")["arr"].shape)
