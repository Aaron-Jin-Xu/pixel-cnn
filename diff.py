import numpy as np
from PIL import Image
from utils import *
from evaluation import *

def tile_plot(imgs, file_path="../plots/test.png", display_size=None):
    if display_size is None:
        s = int(np.sqrt(imgs.shape[0]))
        display_size = (s, s)
    img = Image.fromarray(tile_images(imgs.astype(np.uint8), size=display_size), 'RGB')
    img.save(file_path)


data = np.load("psnr.npz")
all_completed = data['comp']
ground_truth = data['ori']
delta = np.abs(all_completed[0] - all_completed[1])
tile_plot(delta, "test1.png")

#
# delta = np.abs(all_completed[0] - ground_truth)
# tile_plot(delta, "rgbdiff.png")
#
# delta = np.abs(np.mean(all_completed, axis=0) - ground_truth)
# tile_plot(delta, "rgbdiff5.png")

# data = np.load("psnr-gan.npz")
# all_completed = data['blended']
# ground_truth = data['ori']
#
# delta = np.abs(all_completed[0] - all_completed[1])
# tile_plot(delta, "test.png")

# delta = np.abs(all_completed[0] - ground_truth)
# tile_plot(delta, "rgbdiff-gan.png")
#
# delta = np.abs(np.mean(all_completed, axis=0) - ground_truth)
# tile_plot(delta, "rgbdiff5-gan.png")
