# -*- coding: utf-8 -*-
"""
训练基线模型 BaselineCNN。

控制变量对照：与本文方法使用相同输入分辨率、batch size、epoch 数与优化器，
但不使用 BatchNorm、Dropout、数据增强与权重衰减，分类头为 Flatten + 全连接。
产物：
    checkpoints/baseline.pth          模型权重
    results/baseline_history.json     逐 epoch 训练 / 验证曲线
    results/eval_baseline.json        测试集指标、混淆矩阵与各类报告

运行：
    D:\\Anaconda3\\envs\\pyenv\\python.exe train_baseline.py
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.models import BaselineCNN
from src import data as data_utils
from src import trainer


def main():
    config.ensure_dirs()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    print(f"Device: {config.DEVICE}")

    # 基线模型不做任何数据增强，训练 / 验证 / 测试使用同一变换
    eval_tf = data_utils.eval_transform()
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
        data_utils.make_loaders(config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
                                eval_tf, eval_tf, config.BATCH_SIZE)
    class_names = train_ds.classes
    print("Classes:", train_ds.class_to_idx)
    print(f"Train {len(train_ds)} / Val {len(val_ds)} / Test {len(test_ds)}")

    model = BaselineCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = trainer.train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = trainer.evaluate_epoch(
            model, val_loader, criterion, config.DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{config.EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time() - t0:.1f}s)")

    torch.save(model.state_dict(), config.CKPT_BASELINE)

    y_true, y_pred, _ = trainer.predict(model, test_loader, config.DEVICE)
    result = trainer.build_result(
        "BaselineCNN (no BN, no Dropout, no augmentation, Flatten head)",
        len(test_ds), class_names, y_true, y_pred, history=history)

    # 控制台输出分类报告与混淆矩阵具体数值
    trainer.print_report_and_confusion(y_true, y_pred, class_names)

    with open(config.EVAL_BASELINE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(config.HISTORY_BASELINE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nBaseline Test Acc={result['accuracy']:.4f}  "
          f"Macro F1={result['macro_f1']:.4f}")
    print(f"Saved: {config.CKPT_BASELINE}")
    print(f"Saved: {config.EVAL_BASELINE}")
    # print(f"Saved: {config.HISTORY_BASELINE}")


if __name__ == "__main__":
    main()
