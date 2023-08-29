import os
from .coco_detection import CocoDetection
from .nus_wide import NUSWIDE_ZSL
from .pascal_voc import voc2007
from .CIFAR10 import CIFAR10

MODEL_TABLE = {
    'coco': CocoDetection,
    'nus_wide_zsl': NUSWIDE_ZSL,
    'voc2007': voc2007,
    'CIFAR10': CIFAR10
}


def build_dataset(cfg, data_split, annFile=""):
    print(' -------------------- Building Dataset ----------------------')
    print('DATASET.ROOT = %s' % cfg.DATASET.ROOT)
    print('data_split = %s' % data_split)
    print('PARTIAL_PORTION= %f' % cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
    if annFile != "":
        annFile = os.path.join(cfg.DATASET.ROOT, 'annotations', annFile)
    try:
        if 'train' in data_split or 'Train' in data_split:
            img_size = cfg.INPUT.TRAIN.SIZE[0]
        else:
            img_size = cfg.INPUT.TEST.SIZE[0]
    except:
        img_size = cfg.INPUT.SIZE[0]
    print('INPUT.SIZE = %d' % img_size)
    return MODEL_TABLE[cfg.DATASET.NAME](cfg.DATASET.ROOT, data_split, img_size,
                                         p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                                         label_mask=cfg.DATASET.MASK_FILE,
                                         partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)


def build_PLL_dataloaders(cfg, data_split):
    dataset = get_PLL_datasets(cfg, data_split)
    loader = dataset.build_dataloader(batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                                      shuffle=cfg.DATALOADER.TRAIN_X.SHUFFLE)
    return loader


def get_PLL_datasets(cfg, data_split):
    print(' -------------------- Building Dataset ----------------------')
    print('DATASET.ROOT = %s' % cfg.DATASET.ROOT)
    print('data_split = %s' % data_split)
    return MODEL_TABLE[cfg.DATASET.NAME](root=cfg.DATASET.ROOT, 
                                         data_split=data_split,
                                         transform_mean=cfg.INPUT.PIXEL_MEAN,
                                         transform_std=cfg.INPUT.PIXEL_STD,
                                         pp=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)


                                                         
