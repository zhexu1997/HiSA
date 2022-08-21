import logging
from . import datasets as D
from torch.utils.data import DataLoader
from .collate_batch import BatchCollator
from utils.imports import import_file


def build_dataset(dataset_name, dataset_catalog, cfg):
    datasets = []
    data = dataset_catalog.get(dataset_name)
    factory = getattr(D, data["factory"])
    args = data["args"]
    if (cfg.DATASETS.TRAIN == "anet_cap_train" or cfg.DATASETS.TRAIN == "tacos_train") and cfg.FEATURE_TYPE == 'i3d':
        args['features_path'] = args['features_path'].split('.')[0][:-3] + 'i3d.hdf5'
    if cfg.DATASETS.TRAIN == "anet_cap_train" and cfg.FEATURE_TYPE == 'c3d':
        args['features_path'] = args['features_path'].split('.')[0][:-3] + 'c3d.hdf5'
    args['min_count'] = cfg.SENTENCE.MIN_COUNT
    args['train_max_length'] = cfg.SENTENCE.TRAIN_MAX_LENGTH
    args['test_max_length']  = cfg.SENTENCE.TEST_MAX_LENGTH

    dataset = factory(**args)
    return dataset


def make_dataloader(cfg, is_train):
    paths_catalog = import_file("config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train:
        dataset_name = cfg.DATASETS.TRAIN
        dataset = build_dataset(dataset_name,
                                DatasetCatalog,
                                cfg)
        collator = BatchCollator()
        dataloader = DataLoader(dataset,
                                batch_size=cfg.BATCH_SIZE_TRAIN,
                                shuffle=is_train,
                                num_workers=cfg.NUM_WORKERS_TRAIN,
                                collate_fn=collator)
    else:
        dataset_name = cfg.DATASETS.TEST
        dataset = build_dataset(dataset_name,
                                DatasetCatalog,
                                cfg)
        collator = BatchCollator()
        dataloader = DataLoader(dataset,
                                batch_size=cfg.BATCH_SIZE_TEST,
                                shuffle=is_train,
                                num_workers=cfg.NUM_WORKERS_TEST,
                                collate_fn=collator)

    return dataloader, len(dataset)
