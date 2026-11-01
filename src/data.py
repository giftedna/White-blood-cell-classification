# -*- coding: utf-8 -*-
"""
数据集与 DataLoader 公共组件。

目录结构遵循 torchvision.datasets.ImageFolder 的约定：
    images/TRAIN/<类别名>/*.jpg
    images/VAL/<类别名>/*.jpg
    images/TEST/<类别名>/*.jpg
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def train_transform():
    """训练集变换：轻量数据增强 + 归一化（仅本文方法使用）。"""
    return transforms.Compose([
        transforms.Resize((config.IMG_H, config.IMG_W)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def eval_transform():
    """验证 / 测试集变换：仅缩放 + 归一化，不做增强。"""
    return transforms.Compose([
        transforms.Resize((config.IMG_H, config.IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def make_loaders(train_dir, val_dir, test_dir, train_tf, eval_tf,
                 batch_size=config.BATCH_SIZE):
    """构建训练 / 验证 / 测试 DataLoader，并返回训练集类别名。"""
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def class_names(train_dir=config.TRAIN_DIR):
    """仅读取类别名（不构建变换），供评估 / 可视化使用。"""
    return datasets.ImageFolder(train_dir).classes
