# -*- coding: utf-8 -*-
"""
全局配置：路径、超参数、归一化与设备。

所有训练 / 评估 / 可视化 / 出图脚本统一从本模块读取配置，
避免路径和超参数在多个脚本中重复硬编码。
"""
import os
import torch

# ---------------------------------------------------------------------------
# 目录路径（以本文件所在目录为项目根目录）
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(BASE_DIR, "images")
TRAIN_DIR = os.path.join(IMAGES_DIR, "TRAIN")
VAL_DIR = os.path.join(IMAGES_DIR, "VAL")
TEST_DIR = os.path.join(IMAGES_DIR, "TEST")

# 权重目录
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
# 评估结果、训练历史、对照表目录
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# 中文可视化输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# 英文对比配图目录
FIG_DIR = os.path.join(BASE_DIR, "figs")

# 权重文件
CKPT_BASELINE = os.path.join(CHECKPOINT_DIR, "baseline.pth")
CKPT_PROPOSED = os.path.join(CHECKPOINT_DIR, "proposed.pth")

# 评估结果 JSON
EVAL_BASELINE = os.path.join(RESULTS_DIR, "eval_baseline.json")
EVAL_PROPOSED = os.path.join(RESULTS_DIR, "eval_proposed.json")

# 训练历史 JSON
HISTORY_BASELINE = os.path.join(RESULTS_DIR, "baseline_history.json")
HISTORY_PROPOSED = os.path.join(RESULTS_DIR, "proposed_history.json")

# 逐轮训练对照 CSV
HISTORY_CSV = os.path.join(RESULTS_DIR, "training_history_comparison.csv")

# ---------------------------------------------------------------------------
# 训练超参数（两个模型保持一致，保证对比公平）
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 120, 160
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# 图像归一化（mean / std 均为 0.5，映射到 [-1, 1]）
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

NUM_CLASSES = 4

# 类别英文名 -> 中文名（可视化使用）
CLASS_CN = {
    "EOSINOPHIL": "嗜酸性粒细胞",
    "LYMPHOCYTE": "淋巴细胞",
    "MONOCYTE": "单核细胞",
    "NEUTROPHIL": "中性粒细胞",
}

CLASS_ABBR = ["EOS", "LYM", "MON", "NEU"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    """确保所有产物目录存在。"""
    for d in (CHECKPOINT_DIR, RESULTS_DIR, OUTPUT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
