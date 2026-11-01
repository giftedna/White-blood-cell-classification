# -*- coding: utf-8 -*-
"""
统一评估入口：在测试集上评估模型，输出指标 JSON。

模型与权重严格配对，不可互换：
    --model proposed  加载 OptimizedCNN  -> checkpoints/proposed.pth  -> results/eval_proposed.json
    --model baseline  加载 BaselineCNN   -> checkpoints/baseline.pth  -> results/eval_baseline.json
    --model all      依次评估上述两个模型（默认），分别打印各自分类报告与混淆矩阵。

运行示例：
    D:\\Anaconda3\\envs\\pyenv\\python.exe evaluate.py
    D:\\Anaconda3\\envs\\pyenv\\python.exe evaluate.py --model all
    D:\\Anaconda3\\envs\\pyenv\\python.exe evaluate.py --model baseline
    D:\\Anaconda3\\envs\\pyenv\\python.exe evaluate.py --model proposed
"""
import argparse
import json

import torch

import config
from src.models import OptimizedCNN, BaselineCNN
from src import data as data_utils
from src import trainer

# 模型配置映射：模型类、权重路径、结果输出路径、结果标签
MODEL_CONFIG = {
    "proposed": {
        "model_cls": OptimizedCNN,
        "ckpt": config.CKPT_PROPOSED,
        "out": config.EVAL_PROPOSED,
        "tag": "OptimizedCNN (BN + progressive Dropout + augmentation + GAP)",
    },
    "baseline": {
        "model_cls": BaselineCNN,
        "ckpt": config.CKPT_BASELINE,
        "out": config.EVAL_BASELINE,
        "tag": "BaselineCNN (no BN, no Dropout, no augmentation, Flatten head)",
    },
}


def evaluate_one(key, cfg, test_ds, test_loader, class_names):
    """加载单个模型权重，在测试集上预测并打印分类报告与混淆矩阵，写回 JSON。"""
    model = cfg["model_cls"](num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=config.DEVICE))
    model.eval()

    y_true, y_pred, _ = trainer.predict(model, test_loader, config.DEVICE)
    result = trainer.build_result(cfg["tag"], len(test_ds), class_names,
                                  y_true, y_pred)

    # 控制台分模型打印分类报告（precision/recall/f1-score/support）与混淆矩阵数值
    print("\n" + "=" * 72)
    print(f"模型 [{key}]：{cfg['tag']}")
    print("=" * 72)
    trainer.print_report_and_confusion(y_true, y_pred, class_names)

    with open(cfg["out"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {cfg['out']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="白细胞四分类模型测试集评估")
    parser.add_argument("--model", choices=["baseline", "proposed", "all"],
                        default="all", help="选择待评估模型；all 表示依次评估两个模型")
    args = parser.parse_args()

    config.ensure_dirs()

    eval_tf = data_utils.eval_transform()
    _, _, test_ds, _, _, test_loader = data_utils.make_loaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        eval_tf, eval_tf, config.BATCH_SIZE)
    class_names = test_ds.classes

    keys = ["baseline", "proposed"] if args.model == "all" else [args.model]
    results = {}
    for key in keys:
        results[key] = evaluate_one(key, MODEL_CONFIG[key],
                                    test_ds, test_loader, class_names)

    # 两个模型都评估时，在控制台输出整体指标汇总对比
    if len(results) > 1:
        print("\n" + "=" * 72)
        print("测试集整体指标汇总对比")
        print("=" * 72)
        for key, r in results.items():
            print(f"{key:10s} Accuracy={r['accuracy'] * 100:.2f}%  "
                  f"Macro-F1={r['macro_f1'] * 100:.2f}%  "
                  f"Weighted-F1={r['weighted_f1'] * 100:.2f}%")

        b, p = results["baseline"], results["proposed"]
        print(f"\n提升幅度（Proposed - Baseline）："
              f"Accuracy {(p['accuracy'] - b['accuracy']) * 100:+.2f} pp，"
              f"Macro-F1 {(p['macro_f1'] - b['macro_f1']) * 100:+.2f} pp，"
              f"Weighted-F1 {(p['weighted_f1'] - b['weighted_f1']) * 100:+.2f} pp")


if __name__ == "__main__":
    main()
