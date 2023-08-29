import os
from .coco_detection import CocoDetection
from .nus_wide import NUSWIDE_ZSL
from .pascal_voc import voc2007
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch

MODEL_TABLE = {
    'coco': CocoDetection,
    'nus_wide_zsl': NUSWIDE_ZSL,
    'voc2007': voc2007,
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

def build_dataset_PLL(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='RCCC/data/mnist', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='RCCC/data/mnist', train=False, transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='RCCC/data/KMNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.KMNIST(root='RCCC/data/KMNIST', train=False, transform=transforms.ToTensor())
    elif dataname == 'CIFAR10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(), # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ordinary_train_dataset = dsets.CIFAR10(root='/media/zjh/本地磁盘/projects7.12/stingy-teacher/data/data-cifar10', train=True, transform=train_transform, download=True)
        test_dataset = dsets.CIFAR10(root='/media/zjh/本地磁盘/projects7.12/stingy-teacher/data/data-cifar10', train=False, transform=test_transform)
    else:
        raise ValueError "datasets not fuound"  
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=15)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True, num_workers=15)
    num_classes = 10
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes
