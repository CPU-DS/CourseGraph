# -*- coding: utf-8 -*-
# Create Date: 2025/03/24
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/predict.py
# Description: 使用模型进行预测

import os
import torch
from transformers import BertTokenizerFast, BertConfig
from .model import BertBiLSTMCRF
from .config import *

path = os.path.dirname(__file__)


def load_model(model_path):
    config = BertConfig.from_pretrained(model_path)
    model = BertBiLSTMCRF.from_pretrained(model_path, config=config)
    model.eval()
    return model, BertTokenizerFast.from_pretrained(model_path)


def _predict(
    text,
    model,
    tokenizer,
    max_len=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)

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

    entities = []
    char_labels = ["O"] * len(text)
    for i in range(attention_mask.shape[1]):
        if attention_mask[0, i] == 0:
            break
        label = id2label[pred_label_ids[i]]
        offset = offset_mapping[i]
        if label.startswith("B-"):
            char_labels[offset[0]] = label
            char_labels[offset[0]+1: offset[1]] = ["I-" + label[2:]] * (offset[1] - offset[0] - 1)
        elif label.startswith("I-"):
            char_labels[offset[0]: offset[1]] = [label] * (offset[1] - offset[0])

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
            entities.append({"start": start, "end": end, "type": entity_type, "text": text[start:end]})
        else:
            i += 1
    return entities


def batch_predict(texts, model_path, device, max_len=128):
    model, tokenizer = load_model(model_path)

    results = []
    for text in texts:
        entities = _predict(text, model, tokenizer, max_len, device)
        results.append({"text": text, "entities": entities})

    return results


def predict(
    texts: list[str] = None,
    model_path: str = os.path.join(path, "checkpoints/final_model"),
    max_len: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict]:
    """
    使用模型进行预测

    Args:
        texts (list[str]): 需要预测的文本列表
        model_path (str): 模型路径
        max_len (int): 最大长度

    Returns:
        list[dict]: 预测结果列表
    """
    return batch_predict(texts, model_path, device, max_len)
