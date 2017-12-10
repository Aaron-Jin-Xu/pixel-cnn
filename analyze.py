import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
from utils import KL_divergence
plt.style.use("ggplot")
import cv2

def find_coutour(mask):
    contour = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                lower_bound = max(y-1, 0)
                upper_bound = min(y+1, h-1)
                left_bound = max(x-1, 0)
                right_bound = min(x+1, w-1)
                nb = mask[lower_bound:upper_bound+1, left_bound:right_bound+1]
                if np.min(nb)  == 0:
                    contour[y, x] = 1
    return contour

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
    elif t=='mask':
        return ms[image_id, :, :]
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
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(1,1,1)
    ax.imshow(image)
    ax.axis("off")

    # Red channel
    b = 0
    ax = fig.add_subplot(2,2,2)
    ax.plot(np.arange(256), forward_dist[b], label="Forward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, forward_dist[b])))
    ax.plot(np.arange(256), backward_dist[b], label="Backward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, backward_dist[b])))
    ax.plot(np.arange(256), combine_dist[b], label="Combine KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, combine_dist[b])))
    ax.plot(np.arange(256), prior_dist[b], label="Prior KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, prior_dist[b])))
    ax.plot([sample[b]], [0.1], '-o', c='green', markersize=8)
    ax.legend(loc=0)
    ax.set_ylim(0., 0.2)
    ax.set_title("Red Channel")

    # Green channel
    b = 1
    ax = fig.add_subplot(2,2,3)
    ax.plot(np.arange(256), forward_dist[b], label="Forward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, forward_dist[b])))
    ax.plot(np.arange(256), backward_dist[b], label="Backward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, backward_dist[b])))
    ax.plot(np.arange(256), combine_dist[b], label="Combine KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, combine_dist[b])))
    ax.plot(np.arange(256), prior_dist[b], label="Prior KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, prior_dist[b])))
    ax.plot([sample[b]], [0.1], '-o', c='green', markersize=8)
    ax.legend(loc=0)
    ax.set_ylim(0., 0.2)
    ax.set_title("Green Channel")

    # Blue channel
    b = 2
    ax = fig.add_subplot(2,2,4)
    ax.plot(np.arange(256), forward_dist[b], label="Forward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, forward_dist[b])))
    ax.plot(np.arange(256), backward_dist[b], label="Backward KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, backward_dist[b])))
    ax.plot(np.arange(256), combine_dist[b], label="Combine KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, combine_dist[b])))
    ax.plot(np.arange(256), prior_dist[b], label="Prior KL={0:.2f}".format(KL_divergence(combine_dist[b]+1e-10, prior_dist[b])))
    ax.plot([sample[b]], [0.1], '-o', c='green', markersize=8)
    ax.legend(loc=0)
    ax.set_ylim(0., 0.2)
    ax.set_title("Blue Channel")


    fig.savefig("plots/ana-{0}.png".format(str(pid).zfill(4))) #, dpi='figure')
    plt.close()

def make_movie(dir, duration=0.5, name='movie'):
    images = []
    dirpath, dirnames, filenames = next(os.walk(dir))
    for f in filenames:
        if ".png" in f:
            images.append(imageio.imread(os.path.join(dir, f)))
    imageio.mimsave(os.path.join(dir, "{0}.gif".format(name)), images, "GIF", duration=duration)


records = load_records("/Users/Aaron-MAC/Code")
#if not os.path.exists("plots"):
#    os.makedirs("plots")

#analyze_record(records, 0)
#make_movie("plots", 0.5, 'movie-svhn-0')

mask = np.ones((32,32))
mask[10:20, 10:20] = 0

contour = find_coutour(mask)
plt.imshow(contour, cmap='Greys')
plt.axis("off")
plt.show()
