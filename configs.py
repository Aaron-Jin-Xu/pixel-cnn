

configs = {}
configs["svhn-forward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 6,
    "init_batch_size": 6,
}
configs["svhn-backward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-backward",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 6,
    "init_batch_size": 6,
    "masked": True,
    "rot180": True,
}
configs["svhn-backward-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-backward-rename",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 6,
    "init_batch_size": 6,
    "masked": True,
    "rot180": True,
}
