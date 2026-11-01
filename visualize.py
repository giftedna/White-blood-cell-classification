# -*- coding: utf-8 -*-
"""
本文方法（OptimizedCNN）中文可视化，输出到 outputs/ 目录：
    1. training_curves.png       训练 / 验证 Loss 与准确率曲线
    2. confusion_matrix.png      测试集混淆矩阵热力图
    3. classification_report.png 各类别 Precision / Recall / F1 柱状图
    4. predict_samples.png      随机抽样 16 张测试图的预测结果

前置条件：先运行 train_proposed.py 生成权重与训练历史。
运行：
    D:\\Anaconda3\\envs\\pyenv\\python.exe visualize.py
"""
import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import config
from src.models import OptimizedCNN
from src import data as data_utils
from src import trainer

# 中文字体（Windows 使用 SimHei）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training_curves():
    """绘制训练 / 验证曲线；历史文件不存在或为空时跳过。"""
    if not os.path.exists(config.HISTORY_PROPOSED):
        print(f"未找到 {config.HISTORY_PROPOSED}，跳过训练曲线。请先运行 train_proposed.py。")
        return
    history = json.load(open(config.HISTORY_PROPOSED, encoding="utf-8"))
    if not history["train_loss"]:
        print("训练历史为空，跳过训练曲线。请先运行 train_proposed.py。")
        return
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="训练 Loss", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"], label="验证 Loss", marker="s", ms=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss 曲线")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="训练准确率", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"], label="验证准确率", marker="s", ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("准确率曲线")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out}")


def load_model_and_predict():
    """加载权重并在测试集预测，返回标签、概率、类别名与数据集。"""
    if not os.path.exists(config.CKPT_PROPOSED):
        print(f"未找到 {config.CKPT_PROPOSED}，跳过混淆矩阵与样本可视化。")
        return None

    eval_tf = data_utils.eval_transform()
    _, _, test_ds, _, _, test_loader = data_utils.make_loaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        eval_tf, eval_tf, config.BATCH_SIZE)
    class_names = test_ds.classes

    model = OptimizedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CKPT_PROPOSED, map_location=config.DEVICE))
    y_true, y_pred, y_prob = trainer.predict(model, test_loader, config.DEVICE)
    return y_pred, y_true, y_prob, class_names, test_ds


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别"); plt.ylabel("真实类别")
    plt.title(f"混淆矩阵（测试集，准确率 {accuracy_score(y_true, y_pred):.3f}）")
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out}")


def plot_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        vals = [report[c][m] for c in class_names]
        plt.bar(x + i * width, vals, width, label=m)
    plt.xticks(x + width, class_names)
    plt.ylim(0, 1.05); plt.ylabel("分数"); plt.title("各类别 Precision / Recall / F1")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "classification_report.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out}")


def plot_predict_samples(y_true, y_pred, y_prob, class_names, test_ds, n=16):
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(test_ds), size=n, replace=False)
    rows, cols = 4, 4
    plt.figure(figsize=(14, 14))
    for i, k in enumerate(idx):
        img, _ = test_ds[k]
        img = img.numpy().transpose(1, 2, 0) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        true_c = class_names[y_true[k]]
        pred_c = class_names[y_pred[k]]
        conf = y_prob[k].max()
        ok = "√" if y_true[k] == y_pred[k] else "×"
        color = "green" if y_true[k] == y_pred[k] else "red"
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{ok} 真实:{true_c}\n预测:{pred_c} ({conf:.2f})",
                  color=color, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "predict_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out}")


def main():
    config.ensure_dirs()
    print("== 1/4 训练曲线 ==")
    plot_training_curves()

    print("== 2/4 加载模型并预测 ==")
    result = load_model_and_predict()
    if result is None:
        return
    y_pred, y_true, y_prob, class_names, test_ds = result

    print("== 3/4 混淆矩阵 ==")
    plot_confusion_matrix(y_true, y_pred, class_names)

    print("== 4/4 分类报告 + 样本可视化 ==")
    plot_classification_report(y_true, y_pred, class_names)
    plot_predict_samples(y_true, y_pred, y_prob, class_names, test_ds)
    print("\n全部可视化完成。")


if __name__ == "__main__":
    main()
