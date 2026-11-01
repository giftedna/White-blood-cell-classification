# -*- coding: utf-8 -*-
"""
生成英文对比配图，输出到 figs/ 目录：
    fig1_training_curves.png   基线 / 本文方法训练曲线四联对比
    fig2_confusion_matrix.png  两模型测试集混淆矩阵并排对比
    fig3_per_class_metrics.png 各类别 Precision / Recall / F1 分组柱状图
    fig4_architecture.png       本文方法网络结构示意图

数据来源：results/ 下的评估 JSON 与训练历史 JSON。
运行：
    D:\\Anaconda3\\envs\\pyenv\\python.exe make_figs.py
"""
import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import config
from src.models import OptimizedCNN
from src import data as data_utils
from src import trainer

# 英文配图使用英文标签，避免中英文混排问题
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def draw_architecture(fig_dir):
    """绘制本文方法 OptimizedCNN 详细网络结构示意图（fig4）。

    逐层通道数与输出尺寸均与 src/models.py 真实前向探针一致（输入 120×160×3）：
    Block1 60×80×32，Block2 30×40×64，Block3 15×20×128，Block4 7×10×256，
    GAP 4×4 后展平为 4096，分类头 256 → 4 类。
    """
    fig, ax = plt.subplots(figsize=(16.5, 5.4))
    ax.set_xlim(0, 17.6)
    ax.set_ylim(0.5, 5.7)
    ax.axis("off")

    def box(cx, cy, w, h, text, face, txt_color="black", fs=7.4, lw=1.0):
        rect = plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                             facecolor=face, edgecolor="black", lw=lw, zorder=2)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", color=txt_color,
                fontsize=fs, fontweight="bold", zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="black"),
                    zorder=1)

    # ---- 卷积主干（横向流水线）----
    main_y = 3.55
    main_h = 2.1
    box(1.25, main_y, 1.5, main_h, "Input\n\n120×160×3", "#dceafb", fs=8)
    box(3.75, main_y, 2.1, main_h,
        "Conv 3×3, 3→32 · BN · ReLU\nConv 3×3, 32→32 · BN · ReLU\n"
        "MaxPool 2×2, Dropout2d 0.1\n60×80×32", "#bcd6f2", fs=7.0)
    box(6.25, main_y, 2.1, main_h,
        "Conv 3×3, 32→64 · BN · ReLU\nConv 3×3, 64→64 · BN · ReLU\n"
        "MaxPool 2×2, Dropout2d 0.2\n30×40×64", "#8fb9ec", fs=7.0)
    box(8.75, main_y, 2.1, main_h,
        "Conv 3×3, 64→128 · BN · ReLU\nConv 3×3, 128→128 · BN · ReLU\n"
        "MaxPool 2×2, Dropout2d 0.3\n15×20×128", "#5f97df",
        txt_color="white", fs=7.0)
    box(11.25, main_y, 2.1, main_h,
        "Conv 3×3, 128→256 · BN · ReLU\nConv 3×3, 256→256 · BN · ReLU\n"
        "MaxPool 2×2, Dropout2d 0.4\n7×10×256", "#3d76c4",
        txt_color="white", fs=7.0)
    box(13.55, main_y, 1.9, main_h,
        "AdaptiveAvgPool\n4×4\n\n256×4×4", "#cdeccd", fs=7.4)

    main_nodes = [1.25, 3.75, 6.25, 8.75, 11.25, 13.55]
    main_w = [1.5, 2.1, 2.1, 2.1, 2.1, 1.9]
    for i in range(len(main_nodes) - 1):
        arrow(main_nodes[i] + main_w[i] / 2, main_y,
              main_nodes[i + 1] - main_w[i + 1] / 2, main_y)

    # ---- 分类头（右侧纵向堆叠）----
    head_cx = 16.0
    head_w = 2.35
    box(head_cx, 4.75, head_w, 0.9, "Flatten\n4096", "#f7d9a6", fs=8)
    box(head_cx, 3.60, head_w, 0.9, "Dense 256 · BN1d · ReLU",
        "#f2a952", fs=7.6)
    box(head_cx, 2.45, head_w, 0.9, "Dropout 0.5", "#ee8f49", fs=8)
    box(head_cx, 1.30, head_w, 0.9, "Dense 4 · Softmax",
        "#e8635a", txt_color="white", fs=8)
    for cy in (4.75, 3.60, 2.45):
        arrow(head_cx, cy - 0.45, head_cx, cy - 0.75)

    # GAP -> Flatten 折线箭头（竖线走分类头框外间隙，先水平再折向上）
    turn_x = 14.66
    ax.plot([14.5, turn_x], [main_y, main_y], color="black", lw=1.4, zorder=1)
    ax.plot([turn_x, turn_x], [main_y, 4.75], color="black", lw=1.4, zorder=1)
    arrow(turn_x, 4.75, head_cx - head_w / 2, 4.75)

    ax.set_title("Proposed OptimizedCNN architecture", fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_architecture.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig4_architecture.png")


def main():
    config.ensure_dirs()

    with open(config.EVAL_PROPOSED, encoding="utf-8") as f:
        prop = json.load(f)
    with open(config.EVAL_BASELINE, encoding="utf-8") as f:
        base = json.load(f)
    with open(config.HISTORY_PROPOSED, encoding="utf-8") as f:
        prop_hist = json.load(f)
    with open(config.HISTORY_BASELINE, encoding="utf-8") as f:
        base_hist = json.load(f)

    classes = config.CLASS_ABBR  # EOS / LYM / MON / NEU

    # ====== 图1：训练曲线对比（4 子图）======
    if base_hist["train_loss"] and prop_hist["train_loss"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        axes[0, 0].plot(base_hist["train_loss"], label="Train", lw=1.5)
        axes[0, 0].plot(base_hist["val_loss"], label="Val", lw=1.5)
        axes[0, 0].set_title("(a) Baseline: Loss")
        axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(base_hist["train_acc"], label="Train", lw=1.5)
        axes[0, 1].plot(base_hist["val_acc"], label="Val", lw=1.5)
        axes[0, 1].set_title("(b) Baseline: Accuracy")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(prop_hist["train_loss"], label="Train", lw=1.5, color="#1f77b4")
        axes[1, 0].plot(prop_hist["val_loss"], label="Val", lw=1.5, color="#ff7f0e")
        axes[1, 0].set_title("(c) Proposed: Loss")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(prop_hist["train_acc"], label="Train", lw=1.5, color="#1f77b4")
        axes[1, 1].plot(prop_hist["val_acc"], label="Val", lw=1.5, color="#ff7f0e")
        axes[1, 1].set_title("(d) Proposed: Accuracy")
        axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(config.FIG_DIR, "fig1_training_curves.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved fig1_training_curves.png")
    else:
        print("训练历史不完整，跳过 fig1_training_curves.png（请先运行两个训练脚本）。")

    # ====== 图2：混淆矩阵对比 ======
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_b = np.array(base["confusion_matrix"])
    cm_p = np.array(prop["confusion_matrix"])
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[0].set_title(f"(a) Baseline (Acc={base['accuracy']:.3f})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_p, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[1].set_title(f"(b) Proposed (Acc={prop['accuracy']:.3f})")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig2_confusion_matrix.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig2_confusion_matrix.png")

    # ====== 图3：各类别 P/R/F1 对比柱状图 ======
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, m in enumerate(metrics):
        b_vals = [base["per_class"][c][m] for c in prop["class_names"]]
        p_vals = [prop["per_class"][c][m] for c in prop["class_names"]]
        ax.bar(x + i * width, b_vals, width, label=f"Baseline {m}",
               color=plt.cm.Blues(0.3 + i * 0.15))
        ax.bar(x + (i + 3) * width, p_vals, width, label=f"Proposed {m}",
               color=plt.cm.Oranges(0.3 + i * 0.15))
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Score"); ax.set_ylim(0.5, 1.05)
    ax.set_title("Per-class Precision / Recall / F1 comparison")
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig3_per_class_metrics.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig3_per_class_metrics.png")

    # ====== 图4：本文方法 OptimizedCNN 详细网络结构示意图 ======
    draw_architecture(config.FIG_DIR)

    # ====== 图5：四类测试样本示例（每类随机取一张，固定种子保证可复现）======
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    rng = np.random.default_rng(config.SEED)
    for ax, c in zip(axes.flat, prop["class_names"]):
        files = sorted(os.listdir(os.path.join(config.TEST_DIR, c)))
        fn = rng.choice(files)
        img = Image.open(os.path.join(config.TEST_DIR, c, fn)).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(c, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig5_sample_grid.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig5_sample_grid.png")

    # ====== 图6：所提方法测试集预测样本可视化（绿=正确，红=错误）======
    if os.path.exists(config.CKPT_PROPOSED):
        model = OptimizedCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        model.load_state_dict(torch.load(config.CKPT_PROPOSED,
                                         map_location=config.DEVICE))
        model.eval()
        eval_tf = data_utils.eval_transform()
        _, _, test_ds, _, _, test_loader = data_utils.make_loaders(
            config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
            eval_tf, eval_tf, config.BATCH_SIZE)
        class_names = test_ds.classes
        y_true, y_pred, y_prob = trainer.predict(model, test_loader, config.DEVICE)

        idx = rng.choice(len(test_ds), size=16, replace=False)
        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        for ax, k in zip(axes.flat, idx):
            img, _ = test_ds[k]
            img = img.numpy().transpose(1, 2, 0) * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ok = y_true[k] == y_pred[k]
            true_c, pred_c = class_names[y_true[k]], class_names[y_pred[k]]
            conf = y_prob[k].max()
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"True: {true_c}\nPred: {pred_c} ({conf:.2f})",
                         color="green" if ok else "red", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(config.FIG_DIR, "fig6_predict_samples.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved fig6_predict_samples.png")
    else:
        print(f"未找到 {config.CKPT_PROPOSED}，跳过 fig6_predict_samples.png。")

    print("\nAll figures saved to:", config.FIG_DIR)


if __name__ == "__main__":
    main()
