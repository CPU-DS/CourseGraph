# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ke/config.py
# Description: 标签配置


labels = ['IGNORE', 'O', 'B-中心知识点', 'I-中心知识点', 'B-关联知识点', 'I-关联知识点']

label2id = {tag: i for i, tag in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

relations = ['NOTHING', '相关', '顺序', '包含']  # 同义、并列

relation2id = {rel: i for i, rel in enumerate(relations)}
id2relation = {i: rel for rel, i in relation2id.items()}
