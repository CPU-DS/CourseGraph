# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/dataset.py
# Description: NER数据集

import torch
from torch.utils.data import Dataset
from config import *
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
            char_labels[start] = f"B-{type2label[entity_type]}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{type2label[entity_type]}"
        
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            encoding = self.tokenizer(
                text,
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
            
            token_labels = []
            for i, offset in enumerate(offset_mapping):
                if offset[0] == 0 and offset[1] == 0:  # [CLS] 或 [SEP] 或 [PAD]
                    token_labels.append(label2id["O"])
                else:
                    if tokens[i].startswith("##"): # 子词
                        token_labels.append(label2id["X"])
                    else:
                        token_labels.append(label2id[char_labels[offset[0]]])
        else:
            encoding = self.tokenizer(
                text,
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
                    token_labels.append(label2id["O"])
                else:
                    if token.startswith("##"):
                        token = token[2:]
                        token_labels.append(label2id["X"])
                    else:
                        token_labels.append(label2id[char_labels[char_idx]])
                    char_idx += len(token)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [0] * len(input_ids),
            "labels": torch.tensor(token_labels)
        }
