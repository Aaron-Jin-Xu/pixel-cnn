configs = []
configs["svhn-forward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward",
    "nr_filter": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 12,
    "init_batch_size": 100
}
configs["svhn-backward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward",
    "nr_filter": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 12,
    "init_batch_size": 100,
    "masked": True,
    "rot180": True
}
