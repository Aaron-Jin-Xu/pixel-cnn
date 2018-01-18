import numpy as np
from utils import *
from evaluation import *

label = "cross-gan"
data = np.load("psnr-{0}.npz".format(label))
if "gan" in label:
    all_completed = data['blended']
else:
    all_completed = data['comp']
ground_truth = data['ori']

psnr = np.array(batch_psnr(np.mean(all_completed, axis=0), ground_truth, False))
print(psnr.mean())
print(psnr.std(ddof=1) / np.sqrt(psnr.shape[0]))
