# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/vlm.py
# Description: 定义多模态大模型类

from .config import VisualConfig
import torch
from modelscope import AutoModel, AutoTokenizer


class VisualModel(object):

    def __init__(self, path: str,
                 config: VisualConfig = VisualConfig()) -> None:
        """ 多模态大模型

        Args:
            path (str, optional): 模型名称或路径
            config (VisualConfig, optional): 配置. Defaults to VisualConfig().
        """
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True,
            torch_dtype=torch.float16).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.config = config

    def chat(self,
             msgs: list,
             sys_prompt: str = 'You are a helpful assistant.') -> str:
        """ 图片问答

        Args:
            msgs (list): 输入内容
            sys_prompt (str, optional): 系统提示词. Defaults to 'You are a helpful assistant.'.

        Returns:
            str: 模型输出
        """

        return self.model.chat(image=None,
                               msgs=msgs,
                               tokenizer=self.tokenizer,
                               sampling=True,
                               temperature=self.config.temperature,
                               sys_prompt=sys_prompt)
