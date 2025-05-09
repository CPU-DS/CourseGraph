# -*- coding: utf-8 -*-
# Create Date: 2025/04/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/eval.py
# Description: 评估


def evaluate(pred, label):
    # pred/label 为 list of (text, type)
    pred_set = set(pred)
    label_set = set(label)
    correct = len(pred_set & label_set)
    
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(label_set) if label_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    union_size = len(pred_set | label_set)
    acc = correct / union_size if union_size > 0 else 1.0

    return {
        'precision': precision,
        'recall': recall,
        'acc': acc,
        'f1': f1
    }