import numpy as np
import os

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
    forward = get_image_record(records, image_id, t="dis", dist_type="forward")
    backward = get_image_record(records, image_id, t="dis", dist_type="backward")
    combine = get_image_record(records, image_id, t="dis", dist_type="combine")
    sample = get_image_record(records, image_id, t="sample")
    for p in range(num_pixels):
        cur_image = images[p]
        cur_forward_dis = forward[p]
        cur_backward_dis = backward[p]
        cur_combine_dis = combine[p]
        cur_sample = sample[p]
        print(cur_sample)




records = load_records("/data/ziz/jxu")
analyze_record(records, 0)
