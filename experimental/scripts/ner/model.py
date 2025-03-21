# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/model.py
# Description: 模型

import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_path, num_labels, hidden_dim=128, dropout=0.1):
        super(BertBiLSTMCRF, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_dim = self.bert.config.hidden_size
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        bert_output = outputs.last_hidden_state
        bert_output = self.dropout(bert_output)
        
        lstm_output, _ = self.lstm(bert_output)
        lstm_output = self.dropout(lstm_output)
        
        emissions = self.hidden2label(lstm_output)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            pred_labels = self.crf.decode(emissions, mask=attention_mask.bool())
            return pred_labels
