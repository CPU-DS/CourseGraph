# -*- coding: utf-8 -*-
# Create Date: 2025/05/17
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/prompt/metric.py
# Description: 评估指标计算


from typing import Literal
import collections


def _calc_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def compute_sample_metrics(
        pred: list[tuple[str, str]],
        true: list[tuple[str, str]],
        mode: Literal['strict', 'ignore_type', 'per_type'] = 'strict'
):
    """
    计算单个样本的准确率、召回率和F1值

    Args:
        pred (list[tuple[str, str]]): 预测实体 list of (text, type)
        true (list[tuple[str, str]]): 真实实体 list of (text, type)
        mode (Literal['strict', 'ignore_type', 'per_type']): 评估模式, 支持:
            'strict' - 完整匹配 text+type
            'ignore_type' - 只匹配 text
            'per_type' - 按 type 分别计算

    Returns:
        if mode is 'strict' or 'ignore_type', return:
            {'precision': ..., 'recall': ..., 'f1': ...}
        if mode is 'per_type', return:
            {'by_type': {type1: {'precision':..., 'recall':..., 'f1':...}, ...}}  
    """
    if mode in ('strict', 'ignore_type'):
        if mode == 'strict':
            true_set = set(true)
            pred_set = set(pred)
        else:
            true_set = set(text for text, _ in true)
            pred_set = set(text for text, _ in pred)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        return _calc_metrics(tp, fp, fn)

    elif mode == 'per_type':
        types = set(t for _, t in true) | set(t for _, t in pred)
        results = {}
        for t in types:
            true_set = set(text for text, ty in true if ty == t)
            pred_set = set(text for text, ty in pred if ty == t)
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            results[t] = _calc_metrics(tp, fp, fn)
        return {'by_type': results}


def compute_overall_metrics(
        preds: list[list[tuple[str, str]]],
        trues: list[list[tuple[str, str]]],
        mode: Literal['strict', 'ignore_type', 'per_type'] = 'strict'
):
    """
    计算多样本的 Micro 和 Macro 平均指标

    Args:
        preds (list[list[tuple[str, str]]]): 多样本预测 list of list of (text, type)
        trues (list[list[tuple[str, str]]]): 多样本真实 list of list of (text, type)
        mode (Literal['strict', 'ignore_type', 'per_type']): 评估模式, 同 compute_sample_metrics

    返回:
        if mode is 'strict' or 'ignore_type', return:
            {'micro': {'precision':..., 'recall':..., 'f1':...},
             'macro': {'precision':..., 'recall':..., 'f1':...}}
        if mode is 'per_type', return:
            {'micro_by_type': {type1: {...}}, 'macro_by_type': {type1: {...}}}
    """
    if len(preds) != len(trues):
        raise ValueError
    if mode in ('strict', 'ignore_type'):
        global_tp = global_fp = global_fn = 0
        sample_metrics = []
        for p, t in zip(preds, trues):
            m = compute_sample_metrics(p, t, mode)
            sample_metrics.append(m)
            if mode == 'strict':
                true_set = set(t)
                pred_set = set(p)
            else:
                true_set = set(text for text, _ in t)
                pred_set = set(text for text, _ in p)
            global_tp += len(true_set & pred_set)
            global_fp += len(pred_set - true_set)
            global_fn += len(true_set - pred_set)

        micro = _calc_metrics(global_tp, global_fp, global_fn)
        macro_counts = len(sample_metrics)
        macro_tp = sum(m['accuracy'] for m in sample_metrics) / macro_counts
        macro_precision = sum(m['precision'] for m in sample_metrics) / macro_counts
        macro_recall = sum(m['recall'] for m in sample_metrics) / macro_counts
        macro_f1 = sum(m['f1'] for m in sample_metrics) / macro_counts
        macro = {
            'accuracy': macro_tp,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
        return {'micro': micro, 'macro': macro}

    elif mode == 'per_type':
        tp_counts = collections.Counter()
        fp_counts = collections.Counter()
        fn_counts = collections.Counter()
        macro_accum = collections.defaultdict(
            lambda: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'count': 0})

        for p, t in zip(preds, trues):
            sample_by_type = compute_sample_metrics(p, t, mode)['by_type']
            for typ, m in sample_by_type.items():
                macro_accum[typ]['accuracy'] += m['accuracy']
                macro_accum[typ]['precision'] += m['precision']
                macro_accum[typ]['recall'] += m['recall']
                macro_accum[typ]['f1'] += m['f1']
                macro_accum[typ]['count'] += 1
            for text, typ in t:
                fn_counts[typ] += 1
            for text, typ in p:
                fp_counts[typ] += 1
            for typ in set([ty for _, ty in t] + [ty for _, ty in p]):
                true_set = set(txt for txt, ty in t if ty == typ)
                pred_set = set(txt for txt, ty in p if ty == typ)
                tp_counts[typ] += len(true_set & pred_set)

        micro_by_type = {}
        for typ in tp_counts:
            tp = tp_counts[typ]
            fp = fp_counts[typ] - tp
            fn = fn_counts[typ] - tp
            micro_by_type[typ] = _calc_metrics(tp, fp, fn)

        macro_by_type = {}
        for typ, vals in macro_accum.items():
            count = vals['count']
            macro_by_type[typ] = {
                'accuracy': vals['accuracy'] / count,
                'precision': vals['precision'] / count,
                'recall': vals['recall'] / count,
                'f1': vals['f1'] / count
            }

        return {'micro_by_type': micro_by_type, 'macro_by_type': macro_by_type}
