# -*- coding: utf-8 -*-
"""
训练 / 验证 / 测试公共函数，封装损失、准确率、预测与指标统计，
避免在两个训练脚本中重复编写循环代码。
"""
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
)

import config


def train_one_epoch(model, loader, criterion, optimizer, device):
    """执行一个训练 epoch，返回平均损失与准确率。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device):
    """在给定数据集上计算平均损失与准确率（不更新参数）。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def predict(model, loader, device):
    """返回真实标签、预测类别与各类概率（numpy 数组）。"""
    model.eval()
    all_pred, all_true, all_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_prob.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_pred.extend(outputs.argmax(1).cpu().numpy())
            all_true.extend(labels.numpy())
    return np.array(all_true), np.array(all_pred), np.concatenate(all_prob)


def build_result(model_tag, num_test, class_names, y_true, y_pred, history=None):
    """汇总测试集指标，生成与评估脚本一致的结果字典。"""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    result = {
        "model": model_tag,
        "device": str(config.DEVICE),
        "num_test": int(num_test),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
            for c in class_names
        },
    }
    if history is not None:
        result["history"] = history
    return result


def print_report_and_confusion(y_true, y_pred, class_names):
    """在控制台打印分类报告（precision/recall/f1-score/support）与混淆矩阵数值。"""
    # 1) sklearn 分类报告：含各类别指标与 accuracy / macro avg / weighted avg
    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, target_names=class_names,
                               zero_division=0))

    # 2) 混淆矩阵具体数值：行=真实类别，列=预测类别
    cm = confusion_matrix(y_true, y_pred)
    name_w = max(len(c) for c in class_names)
    col_w = max(name_w, 4) + 2
    print("=== 混淆矩阵（行=真实类别，列=预测类别）===")
    header = " " * (name_w + 2) + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, c in enumerate(class_names):
        row = f"{c:<{name_w + 2}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
        print(row)
