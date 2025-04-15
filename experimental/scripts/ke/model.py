# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/model.py
# Description: 模型

import torch.nn as nn
from transformers import BertModel, PreTrainedModel
from torchcrf import CRF
import torch
from torch.nn import functional as F


class BertBiLSTMCRF(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(config._name_or_path)
        self.bert_dim = self.bert.config.hidden_size
        self.hidden_dim = 128
        
        self.lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=self.hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.hidden2label = nn.Linear(self.hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, token_type_ids, valid_mask, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        bert_output = outputs.last_hidden_state
        bert_output = self.dropout(bert_output)  # (batch_size, max_len, hidden_size)
        
        lstm_output, _ = self.lstm(bert_output)
        lstm_output = self.dropout(lstm_output)
        
        emissions = self.hidden2label(lstm_output)  # (batch_size, max_len, num_labels)
        
        pred_label_ids = self.crf.decode(emissions, mask=valid_mask.bool())

        batch_size, seq_len = valid_mask.shape
        padded_preds = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # ignore_index
        flat_preds = torch.cat([torch.tensor(seq, dtype=torch.long) for seq in pred_label_ids])  # flatten
        valid_indices = valid_mask.view(-1).nonzero(as_tuple=False).squeeze()  # valid indices
        padded_preds.view(-1)[valid_indices] = flat_preds  # fill

        pred_label_ids = padded_preds
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=valid_mask.bool(), reduction='mean')
            return {
                "loss": loss,
                "pred_label_ids": pred_label_ids
            }

        return {
            "pred_label_ids": pred_label_ids
        }


class BertForRE(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_relations = config.num_relations
        self.bert = BertModel.from_pretrained(config._name_or_path)
        self.bert_dim = self.bert.config.hidden_size
        
        self.entity_fc = nn.Linear(self.bert_dim, self.bert_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_relations)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        e1_h = self.entity_fc(_entity_average(sequence_output, e1_mask))  # (batch_size, hidden_size)
        e2_h = self.entity_fc(_entity_average(sequence_output, e2_mask))
        
        concat_h = torch.cat([e1_h, e2_h], dim=-1)  # (batch_size, hidden_size*2)
        concat_h = self.dropout(concat_h)
        logits = self.classifier(concat_h)  # (batch_size, num_relations)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_relations), labels.view(-1))
            
            return {
                "loss": loss,
                "logits": logits
            }
        
        return {
            "logits": logits
        }


def _entity_average(hidden_output, e_mask):  # 实体平均向量表示
    e_mask_unsqueeze = e_mask.unsqueeze(2)  # (batch_size, seq_len, 1)
    length = (e_mask != 0).sum(dim=1).unsqueeze(1)  # (batch_size, 1)
    length = torch.clamp(length, min=1)  # 防止除0错误

    sum_vector = torch.sum(hidden_output * e_mask_unsqueeze, dim=1)  # (batch_size, hidden_size)
    avg_vector = sum_vector / length.float()  # (batch_size, hidden_size)
    
    return avg_vector
