import numpy as np
import os

def load_records(dir):
    path = os.path.join(dir, "inpainting_record.npz")
    d = np.load(path)
    return d['img'], d['dis'], d['smp']

def get_image_record(records, image_id, t="image", dist_type="combine"):
    img, dis, smp = records
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
