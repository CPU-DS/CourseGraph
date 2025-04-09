# -*- coding: utf-8 -*-
# Create Date: 2025/03/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/predict.py
# Description: 使用模型进行预测

import os
import torch
from transformers import BertTokenizerFast, BertConfig
from .model import BertBiLSTMCRF, BertForRE
from .config import *
from .dataset import _create_entity_mask


def load_model(model_path, mode):
    config = BertConfig.from_pretrained(model_path)
    ModelClass = BertBiLSTMCRF if mode == "ner" else BertForRE
    model = ModelClass.from_pretrained(model_path, config=config)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(model_path)  # only support fast tokenizer
    return model, tokenizer


def _ner_predict(
        text,
        model,
        tokenizer,
        max_len,
        device,
        exceed_strategy
):
    model.to(device)

    pred_label_ids = []

    if exceed_strategy == "truncation":
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = torch.zeros_like(input_ids).to(device)
        offset_mapping = encoding["offset_mapping"].squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        pred_label_ids = outputs["pred_label_ids"].cpu().numpy()[0]

    else:  # exceed_strategy == "sliding_window":
        full_encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = full_encoding["input_ids"].to(device)
        tokens = full_encoding.tokens()
        attention_mask = full_encoding["attention_mask"].to(device)
        token_type_ids = torch.zeros_like(input_ids).to(device)
        offset_mapping = full_encoding["offset_mapping"].squeeze().tolist()

        if len(tokens) <= max_len:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            pred_label_ids = outputs["pred_label_ids"].cpu().numpy()[0]

        else:
            window_size = max_len
            stride = window_size // 2

            start_token_idx = 0
            while True:
                end_token_idx = min(start_token_idx + window_size, len(tokens))
                with torch.no_grad():
                    window_pred_label_ids = model(
                        input_ids=input_ids[start_token_idx:end_token_idx],
                        attention_mask=attention_mask[start_token_idx:end_token_idx],
                        token_type_ids=token_type_ids[start_token_idx:end_token_idx],
                    )["pred_label_ids"].cpu().numpy()[0]
                if end_token_idx >= len(tokens):
                    pred_label_ids.extend(window_pred_label_ids)  # 最后一个窗口全部保留
                    break
                else:
                    pred_label_ids.extend(window_pred_label_ids[0: stride])  # 只保留每个窗口的前 stride 部分

                start_token_idx += stride

    entities = []
    char_labels = ["O"] * len(text)
    for i in range(attention_mask.shape[1]):
        if attention_mask[0, i] == 0:
            break
        label = id2label[pred_label_ids[i]]
        offset = offset_mapping[i]
        if label.startswith("B-"):
            char_labels[offset[0]] = label
            char_labels[offset[0] + 1: offset[1]] = ["I-" + label[2:]] * (offset[1] - offset[0] - 1)
        elif label.startswith("I-"):
            char_labels[offset[0]: offset[1]] = [label] * (offset[1] - offset[0])

    # 从 char_labels 中推断实体
    i = 0
    while i < len(char_labels):
        if char_labels[i].startswith("B-"):
            entity_type = char_labels[i][2:]
            start = i
            i += 1
            while i < len(char_labels) and (
                    char_labels[i] == f"I-{entity_type}"
                    or (char_labels[i] == f"O" and tokens[i].startswith("##"))
            ):
                i += 1
            end = i
            entities.append({"start": start,
                             "end": end,
                             "type": entity_type,
                             "text": text[start:end]})  # 缺少 id
        else:
            i += 1
    return entities


def _re_predict(
        text,
        e1,
        e2,
        model,
        tokenizer,
        max_len,
        device
):
    model.to(device)

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids).to(device)
    offset_mapping = encoding["offset_mapping"].squeeze().tolist()

    e1_mask = _create_entity_mask(input_ids, offset_mapping, e1[0], e1[1])
    e2_mask = _create_entity_mask(input_ids, offset_mapping, e2[0], e2[1])

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            e1_mask=e1_mask,
            e2_mask=e2_mask
        )

    logits = outputs["logits"]
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = logits.argmax(dim=1).item()
    relation = id2relation[pred_idx]
    probability = probs[pred_idx]
    return {"source": text[e1[0]:e1[1]],
            "target": text[e2[0]:e2[1]],
            "type": relation,
            "probability": float(probability)}  # 缺少 source_id 和 target_id, 多 probability


def ner_predict(
        text: str,
        model_path: str = "experimental/scripts/ke/checkpoints/ner/final_model",
        max_len: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        exceed_strategy: str = "truncation"
) -> list[dict]:
    """
    使用模型进行实体预测

    Args:
        texts (list[str]): 需要预测的文本列表
        model_path (str): 模型路径. Defaults to "experimental/scripts/ke/checkpoints/ner/final_model".
        max_len (int): 最大长度. Defaults to 512.
        device (str): 设备. Defaults to "cuda" or "cpu".

    Returns:
        list[dict]: 预测结果列表
    """
    model, tokenizer = load_model(model_path, "ner")
    return _ner_predict(text, model, tokenizer, max_len, device, exceed_strategy)


def re_predict(
        text: str,
        e1_range: tuple[int, int],
        e2_range: tuple[int, int],
        model_path: str = "experimental/scripts/ke/checkpoints/re/final_model",
        max_len: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        exceed_strategy: str = "truncation"
) -> dict:
    """
    使用模型进行关系预测

    Args:
        text (str): 需要预测的文本
        e1_range (tuple[int, int]): 实体1的位置 [start:end]
        e2_range (tuple[int, int]): 实体2的位置 [start:end]
        model_path (str): 模型路径. Defaults to "experimental/scripts/ke/checkpoints/re/final_model".
        max_len (int): 最大长度. Defaults to 512.
        device (str): 设备. Defaults to "cuda" or "cpu".

    Returns:
        dict: 预测结果
    """
    model, tokenizer = load_model(model_path, "re")
    return _re_predict(text, e1_range, e2_range, model, tokenizer, max_len, device, exceed_strategy)
