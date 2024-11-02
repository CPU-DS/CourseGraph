# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/vlm.py
# Description: 定义图文理解模型类

from .config import vlm_config
import torch
from modelscope import AutoModel, AutoTokenizer


class VLM:

    def __init__(self, path: str) -> None:
        """ 图文理解模型

        Args:
            path (str, optional): 模型名称或路径
        """
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True,
            torch_dtype=torch.float16).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.instruction = 'You are a helpful assistant.'

    def chat(self, msgs: list[dict]) -> str:
        """ 图片问答

        Args:
            msgs (list): 输入内容

        Returns:
            str: 模型输出
        """
        return self.model.chat(image=None,
                               msgs=msgs,
                               tokenizer=self.tokenizer,
                               sampling=True,
                               temperature=vlm_config.temperature,
                               sys_prompt=self.instruction)
