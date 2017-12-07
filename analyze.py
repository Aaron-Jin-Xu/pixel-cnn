import numpy as np

def load_record(dir):
    path = os.path.join(dir, "inpainting_record.npz")
    return os.load(path)

def read_record(image_id):
    pass

def current_state(record, cur_image, next_dis):
    pass
