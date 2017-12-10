import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from utils import KL_divergence
plt.style.use("ggplot")
import cv2

def find_coutour(mask):
    border = cv2.copyMakeBorder((mask*255).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
    _, contours, hierarchy = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    return contours[0]

def load_records(dir):
    path = os.path.join(dir, "inpainting_record.npz")
    d = np.load(path)
    params = {}
    params['num_images'] = d['dis'].shape[3]
    params['num_pixels'] = d['dis'].shape[0]
    return d['img'].astype(np.uint8), d['dis'], d['smp'], d['ms'], params

def get_image_record(records, image_id, t="image", dist_type="combine"):
    img, dis, smp, ms, params = records
    if t=='image':
        return img[:, image_id, :, :, :]
    elif t=='dist':
        if dist_type=='forward':
            return dis[:, :, 0, image_id, :]
        elif dist_type=='backward':
            return dis[:, :, 1, image_id, :]
        elif dist_type=='combine':
            return dis[:, :, 2, image_id, :]
        elif dist_type=='prior':
            return dis[:, :, 3, image_id, :]
        else:
            raise Exception(t+" type not found")
    elif t=='sample':
        return smp[:, image_id, :]
    else:
        raise Exception(t+" type not found")

def analyze_record(records, image_id):

    _, _, _, _, params = records
    num_images = params['num_images']
    assert image_id < num_images, "image_id too large"
    num_pixels = params['num_pixels']
    images = get_image_record(records, image_id, t="image")
    forward = get_image_record(records, image_id, t="dist", dist_type="forward")
    backward = get_image_record(records, image_id, t="dist", dist_type="backward")
    combine = get_image_record(records, image_id, t="dist", dist_type="combine")
    prior = get_image_record(records, image_id, t="dist", dist_type="prior")
    sample = get_image_record(records, image_id, t="sample")
    for p in range(num_pixels):
        cur_image = images[p]
        cur_forward_dis = forward[p]
        cur_backward_dis = backward[p]
        cur_combine_dis = combine[p]
        cur_prior_dis = prior[p]
        cur_sample = sample[p]
        plot(cur_forward_dis, cur_backward_dis, cur_combine_dis, cur_prior_dis, cur_image, cur_sample, pid=p)


def plot(forward_dist, backward_dist, combine_dist, prior_dist, image, sample, pid):
    fig = plt.figure(figsize=(16,16))

    ax = fig.add_subplot(2,2,1)
    ax.imshow(image)
    ax.axis("off")

    # Red channel
    ax = fig.add_subplot(2,2,2)
    ax.plot(np.arange(256), forward_dist[0], label="Forward KL={0:.2f}".format(KL_divergence(forward_dist[0], combine_dist[0]+1e-10)))
    ax.plot(np.arange(256), backward_dist[0], label="Backward KL={0:.2f}".format(KL_divergence(backward_dist[0], combine_dist[0]+1e-10)))
    ax.plot(np.arange(256), combine_dist[0], label="Combine KL={0:.2f}".format(KL_divergence(combine_dist[0], combine_dist[0]+1e-10)))
    ax.plot(np.arange(256), prior_dist[0], label="Prior KL={0:.2f}".format(KL_divergence(prior_dist[0], combine_dist[0]+1e-10)))
    ax.plot([sample[0]], [0.1], '-o', c='green', markersize=8)
    ax.legend(loc=0)
    ax.set_ylim(0., 0.2)
    ax.set_title("Red Channel")
    fig.savefig("plots/ana-{0}.png".format(str(pid).zfill(4)))
    plt.close()

def make_movie(dir, duration=0.5):
    images = []
    dirpath, dirnames, filenames = next(os.walk(dir))
    for f in filenames:
        if ".png" in f:
            images.append(imageio.imread(os.path.join(dir, f)))
    imageio.mimsave(os.path.join(dir, "movie.gif"), images, "GIF", duration=duration)





records = load_records("/Users/Aaron-MAC/Code")
if not os.path.exists("plots"):
    os.makedirs("plots")

analyze_record(records, 0)
make_movie("plots", 0.5)
