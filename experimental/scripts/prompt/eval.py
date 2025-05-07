# -*- coding: utf-8 -*-
# Create Date: 2025/04/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/eval.py
# Description: 评估


def evaluate(pred, label):
    # pred/label 为 list of (text, type)
    correct = len(set(pred) & set(label))
    pred_total = len(pred)
    precision = correct / pred_total if pred_total > 0 else 0.0
    label_total = len(label)
    recall = correct / label_total if label_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
