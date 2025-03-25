# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/dataset.py
# Description: 数据集

import torch
from torch.utils.data import Dataset
from .config import *
from transformers import PreTrainedTokenizerFast


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        entities = self.data[idx]['entities']
        char_labels = ["O"] * len(text)  # 原始标记是字符级

        for e in entities:
            start, end = e["start"], e["end"]
            entity_type = e["type"]
            char_labels[start] = f"B-{entity_type}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{entity_type}"

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            offset_mapping = encoding["offset_mapping"].squeeze().tolist()

            # 从实体得到 token_labels
            token_labels = []
            for i, offset in enumerate(offset_mapping):
                token = tokens[i]
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    token_labels.append(label2id["IGNORE"])
                else:
                    if token.startswith("##"):
                        token_labels.append(label2id["IGNORE"])
                    else:
                        token_labels.append(label2id[char_labels[offset[0]]])
        else:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            token_labels = []
            char_idx = 0
            for token in tokens:
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    token_labels.append(label2id["IGNORE"])
                else:
                    if token.startswith("##"):
                        token = token[2:]
                        token_labels.append(label2id["IGNORE"])
                    else:
                        token_labels.append(label2id[char_labels[char_idx]])
                    char_idx += len(token)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
            "labels": torch.tensor(token_labels)
        }


class REDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data = []  # 每条数据只保留一个关系
        for line in data:
            text = line['text']
            entities = line['entities']
            relations = line['relations']
            for relation in relations:
                self.data.append({
                    'text': text,
                    'e1': next(filter(lambda x: x['id'] == relation['source_id'], entities)),
                    'e2': next(filter(lambda x: x['id'] == relation['target_id'], entities)),
                    'relation': relation['type']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        e1 = item['e1']
        e2 = item['e2']
        relation = item['relation']

        e1_start, e1_end = e1["start"], e1["end"]
        e2_start, e2_end = e2["start"], e2["end"]

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            offset_mapping = encoding["offset_mapping"].squeeze().tolist()

            e1_mask = _create_entity_mask(input_ids, offset_mapping, e1_start, e1_end)
            e2_mask = _create_entity_mask(input_ids, offset_mapping, e2_start, e2_end)

        else:
            
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            e1_mask = _create_entity_mask2(text, input_ids, tokens, e1_start, e1_end)
            e2_mask = _create_entity_mask2(text, input_ids, tokens, e2_start, e2_end)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "labels": torch.tensor(relation2id[relation])
        }


def _create_entity_mask(input_ids, offset_mapping, start, end):
    mask = torch.zeros_like(input_ids)
    for i, offset in enumerate(offset_mapping):
        if offset[0] >= start and offset[1] <= end:
            mask[i] = 1
        elif offset[0] > end:
            break
    return mask


def _create_entity_mask2(text, input_ids, tokens, start, end):
    char_idx = 0
    mask = torch.zeros_like(input_ids)
    while char_idx < len(text):
        token = tokens[char_idx]
        if token.startswith("##"):  # 无需处理PAD
            token = token[2:]
        next_idx = char_idx + len(token)
        if char_idx >= start and next_idx <= end:
            mask[char_idx:next_idx] = 1
        elif char_idx >= end:
            break
        char_idx = next_idx
    return mask
