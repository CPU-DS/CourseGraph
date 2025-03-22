# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/model.py
# Description: 模型

import torch.nn as nn
from transformers import BertModel, PreTrainedModel
from torchcrf import CRF
import torch
from torch.nn import functional as F


class BertBiLSTMCRF(PreTrainedModel):
    def __init__(self, config):
        super(BertBiLSTMCRF, self).__init__(config)
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
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
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
        
        pred_label_ids = self.crf.decode(emissions, mask=attention_mask.bool())
        pred_label_ids = torch.tensor(pred_label_ids, device=emissions.device)
        pred_label_ids[pred_label_ids == 0] = 1 # 模型后处理，不允许预测出现 IGNORE
        pred_label_ids = F.pad(pred_label_ids, (0, labels.shape[1] - pred_label_ids.shape[1]), value=0, mode="constant")
        
        if labels is not None:
            valid_mask = labels != 0
            loss = -self.crf(emissions, labels, mask=valid_mask)
            return {
                "loss": loss,
                "pred_label_ids": pred_label_ids
            }
        else:
            return {
                "pred_label_ids": pred_label_ids
            }