# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/dataset.py
# Description: 数据集

import torch
from torch.utils.data import Dataset
from config import *
from transformers import PreTrainedTokenizerFast


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, **config):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        if config["overflow_strategy"] == "sliding_window":
            # only support fast tokenizer temporarily
            for item in data:
                text = item['text']
                entities = item['entities']

                full_encoding = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_tensors="pt"
                )

                tokens = full_encoding.tokens()

                if len(tokens) <= max_len:
                    item['encoding'] = full_encoding
                    self.data.append(item)
                    continue
                
                offset_mapping = full_encoding["offset_mapping"].squeeze().tolist()
                window_size = max_len
                stride = window_size // 2

                start_token_idx = 0
                while start_token_idx < len(tokens):
                    end_token_idx = min(start_token_idx + window_size, len(tokens))

                    # [start_token_idx, end_token_idx) ==> [start_char_idx, end_char_idx)
                    start_char_idx = offset_mapping[start_token_idx][0]
                    end_char_idx = offset_mapping[end_token_idx - 1][1]

                    # 对每一个窗口, 只保留完全在当前窗口内的实体 (可能会减少窗口长度)
                    for entity in entities:
                        # bais: 实体长度远低于 window_size 和 stride
                        if entity['start'] <= start_char_idx < entity['end']:
                            start_char_idx = entity['end']
                        elif entity['start'] <= end_char_idx < entity['end']:
                            end_char_idx = entity['start']
                            break
                        # start_char_idx 和 end_char_idx 也应该变化，但这里不处理
                        elif entity['start'] > end_char_idx:  # 假定 entity 有序的
                            break

                    window_entities = []
                    for entity in entities:
                        if entity['start'] >= start_char_idx and entity['end'] <= end_char_idx:
                            new_entity = entity.copy()
                            new_entity['start'] -= start_char_idx
                            new_entity['end'] -= start_char_idx
                            window_entities.append(new_entity)

                    window_text = text[start_char_idx:end_char_idx]
                    window_data = {
                        'text': window_text,
                        'entities': window_entities  # 暂时不添加 encoding
                    }
                    self.data.append(window_data)

                    next_token_idx = start_token_idx + stride  # 重叠窗口
                    start_token_idx = next_token_idx
        else:  # default truncation
            self.data = data

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

        # char_labels 对齐为 token_labels
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            if self.data[idx].get('encoding'):
                encoding = self.data[idx]['encoding']  # 预处理阶段可能得到
            else:
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
            tokens = encoding.tokens()
            offset_mapping = encoding["offset_mapping"].squeeze().tolist()  # 每个 token 在原文中的位置

            # 从实体得到 token_labels
            token_labels = ["O"] * len(tokens)
            valid_mask = torch.ones_like(input_ids)
            for i, offset in enumerate(offset_mapping):
                token = tokens[i]
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    valid_mask[i] = 0
                else:
                    if token.startswith("##"):
                        valid_mask[i] = 0
                    else:
                        token_labels[i] = char_labels[offset[0]]
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
            tokens = encoding.tokens()

            token_labels = ["O"] * len(tokens)
            valid_mask = torch.ones_like(input_ids)
            char_idx = 0
            for i, token in enumerate(tokens):
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    valid_mask[i] = 0
                else:
                    if token.startswith("##"):
                        valid_mask[i] = 0
                        token = token[2:]
                    else:
                        token_labels[i] = char_labels[char_idx]
                    char_idx += len(token)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
            "valid_mask": valid_mask,
            "labels": torch.tensor([ label2id[label] for label in  token_labels])
        }


class REDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, overflow_strategy="truncation"):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data = []  # 每条数据只保留一个关系
        for line in data:
            text = line['text']
            entities = line['entities']
            relations = line['relations']
            
            if overflow_strategy == "truncation":
                for relation in relations:
                    self.data.append({
                        'text': text,
                        'e1': next(filter(lambda x: x['id'] == relation['source_id'], entities)),
                        'e2': next(filter(lambda x: x['id'] == relation['target_id'], entities)),
                        'relation': relation['type']
                    })
            elif overflow_strategy == "sliding_window":
                pass
            else:
                pass

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

            e1_mask = _create_entity_mask(input_ids, offset_mapping, e1_start, e1_end)  # 实体掩码为 1
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
