import numpy as np
from PIL import Image
from data.celeba_data import read_imgs

#imgs = read_imgs("/data/ziz/not-backed-up/jxu/CelebA/celeba64-train")
imgs= np.load("/data/ziz/not-backed-up/jxu/SVHN/SVHN32-train.npz")['arr']

num_samples = imgs.shape[0]
p = np.array([(imgs==i).astype(np.uint32).astype(np.float32).sum(0) for i in range(256)])
p = (p+1) / num_samples
np.savez_compressed("/data/ziz/jxu/prior-svhn", arr=p)

print(np.load("/data/ziz/jxu/prior-svhn.npz")["arr"].shape)
