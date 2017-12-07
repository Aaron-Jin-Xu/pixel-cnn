import numpy as np
import os
import matplotlib.pyplot as plt

def load_records(dir):
    path = os.path.join(dir, "inpainting_record.npz")
    d = np.load(path)
    params = {}
    params['num_images'] = d['dis'].shape[0]
    params['num_pixels'] = d['dis'].shape[1]
    return d['img'], d['dis'], d['smp'], params

def get_image_record(records, image_id, t="image", dist_type="combine"):
    img, dis, smp, params = records
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


    _, _, _, params = records
    num_images = params['num_images']
    assert image_id < num_images, "image_id too large"
    num_pixels = params['num_pixels']
    images = get_image_record(records, image_id, t="image")
    forward = get_image_record(records, image_id, t="dist", dist_type="forward")
    backward = get_image_record(records, image_id, t="dist", dist_type="backward")
    combine = get_image_record(records, image_id, t="dist", dist_type="combine")
    sample = get_image_record(records, image_id, t="sample")
    for p in range(num_pixels):
        cur_image = images[p]
        cur_forward_dis = forward[p]
        cur_backward_dis = backward[p]
        cur_combine_dis = combine[p]
        cur_sample = sample[p]

        plot(cur_forward_dis, cur_backward_dis, cur_combine_dis, cur_image, cur_sample)
        break

        print("red", cur_forward_dis[0])
        print("red", cur_backward_dis[0])
        print("red", cur_combine_dis[0])
        print("sample", cur_sample[0])
        print("------------------------")


def plot(forward_dist, backward_dist, combine_dist, image, sample):
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(2,2,1)
    ax.imshow(image)
    ax.axis("off")

    # Red channel
    ax = fig.add_subplot(2,2,2)
    ax.plot(np.arange(256), forward_dist[0], label="forward")
    ax.plot(np.arange(256), backward_dist[0], label="backward")
    ax.plot(np.arange(256), combine_dist[0], label="combine")

    ax.legend(loc=0)
    ax.set_ylim(0., 0.15)
    ax.set_title("Red Channel")
    fig.savefig("analyze_report.png")
    plt.close()









records = load_records("/data/ziz/jxu")
analyze_record(records, 0)
