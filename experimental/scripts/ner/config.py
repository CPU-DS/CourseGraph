# -*- coding: utf-8 -*-
# Create Date: 2025/03/21
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: experimental/scripts/ner/config.py
# Description: NER配置

type2label = {'中心知识点': 'A', '关联知识点': 'B'}
labels = ['O', 'B-A', 'I-A', 'B-B', 'I-B', 'X']

label2id = {tag: i for i, tag in enumerate(labels)}
label2id["X"] = -100  # 子词标签单独设置

id2label = {i: label for label, i in label2id.items()}
