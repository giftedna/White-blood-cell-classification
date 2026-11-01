# -*- coding: utf-8 -*-
"""
导出基线与本文方法逐 epoch 训练对照数据。

产物：results/training_history_comparison.csv
同时在控制台打印两个模型的训练过程 Markdown 表格与关键统计。
运行：
    D:\\Anaconda3\\envs\\pyenv\\python.exe print_history.py
"""
import json
import csv

import config


def load_history(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    base = load_history(config.HISTORY_BASELINE)
    prop = load_history(config.HISTORY_PROPOSED)

    # 写出对照表 CSV
    with open(config.HISTORY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Baseline_Train_Loss", "Baseline_Train_Acc",
            "Baseline_Val_Loss", "Baseline_Val_Acc",
            "Proposed_Train_Loss", "Proposed_Train_Acc",
            "Proposed_Val_Loss", "Proposed_Val_Acc",
        ])
        n_epochs = min(len(base["train_loss"]), len(prop["train_loss"]))
        for i in range(n_epochs):
            writer.writerow([
                i + 1,
                round(base["train_loss"][i], 4), round(base["train_acc"][i], 4),
                round(base["val_loss"][i], 4), round(base["val_acc"][i], 4),
                round(prop["train_loss"][i], 4), round(prop["train_acc"][i], 4),
                round(prop["val_loss"][i], 4), round(prop["val_acc"][i], 4),
            ])
    print(f"CSV saved: {config.HISTORY_CSV}\n")

    for title, history in [("BaselineCNN", base), ("Proposed OptimizedCNN", prop)]:
        print(f"## {title} 训练过程（{len(history['train_loss'])} epochs）\n")
        print("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |")
        print("|---|---|---|---|---|")
        for i in range(len(history["train_loss"])):
            print(f"| {i + 1} | {history['train_loss'][i]:.4f} | "
                  f"{history['train_acc'][i]:.4f} | "
                  f"{history['val_loss'][i]:.4f} | "
                  f"{history['val_acc'][i]:.4f} |")
        print()

    print("## 关键统计\n")
    for title, history in [("Baseline", base), ("Proposed", prop)]:
        if not history["val_acc"]:
            print(f"{title}: 历史数据为空，请先运行对应训练脚本。")
            continue
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        print(f"{title}: 最佳验证准确率={best_val_acc:.4f} (Epoch {best_epoch}), "
              f"最终训练准确率={history['train_acc'][-1]:.4f}, "
              f"最终验证准确率={history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
