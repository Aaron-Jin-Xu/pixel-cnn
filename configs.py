

configs = {}

## SVHN
configs["svhn-forward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward",
    #"save_dir": "/data/ziz/jxu/save-svhn-forward-less-epoch",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "nr_gpu": 8,
}
configs["svhn-backward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-backward",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "nr_gpu": 8,
}
configs["svhn-backward-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-backward-rename",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "nr_gpu": 8,
}

## CelebA
configs["celeba-forward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save-forward",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "celeba",
    "batch_size": 6,
    "init_batch_size": 6,
    "nr_gpu": 8,
}
configs["celeba-backward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save-backward",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "celeba",
    "batch_size": 6,
    "init_batch_size": 6,
    "masked": True,
    "rot180": True,
    "nr_gpu": 8,
}
configs["celeba-backward-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save-backward-rename",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "celeba",
    "batch_size": 6,
    "init_batch_size": 6,
    "masked": True,
    "rot180": True,
    "nr_gpu": 8,
}

configs["celeba-hr-forward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-forward",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "save_interval":5,
    "nr_gpu":8,
}

configs["celeba-hr-backward"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-backward",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "save_interval":5,
    "nr_gpu":8,
}

configs["celeba-hr-backward-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-backward-rename",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "save_interval":5,
    "nr_gpu":8,
}

configs["celeba-hr-test"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-forward-new-20-e60",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "save_interval":5,
    "nr_gpu":8,
    "nr_logistic_mix": 20,
}

configs["celeba-hr-forward-new-20"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-forward-new-20-e40",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "save_interval":5,
    "nr_gpu":8,
    "nr_logistic_mix": 20,
}
configs["celeba-hr-forward-new-20-missing"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-forward-new-20-missing",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "save_interval":5,
    "nr_gpu":8,
    "nr_logistic_mix": 20,
    "masked": True,
}

configs["celeba-hr-backward-new-20-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save64-backward-new-20-rename",
    "nr_filters": 100,
    "nr_resnet": 4,
    "data_set": "celeba",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "save_interval":5,
    "nr_gpu":8,
    'nr_logistic_mix': 20,
}


configs["svhn-forward-20"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward-20",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "nr_gpu": 8,
    "nr_logistic_mix": 20,
}
configs["svhn-forward-20-missing"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-forward-20-missing",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "nr_gpu": 8,
    "nr_logistic_mix": 20,
    "masked": True,
}
configs["svhn-backward-20-rename"] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
    "save_dir": "/data/ziz/jxu/save-svhn-backward-20-rename",
    "nr_filters": 100,
    "nr_resnet": 5,
    "data_set": "svhn",
    "batch_size": 8,
    "init_batch_size": 8,
    "masked": True,
    "rot180": True,
    "nr_gpu": 8,
    "nr_logistic_mix": 20,
}
