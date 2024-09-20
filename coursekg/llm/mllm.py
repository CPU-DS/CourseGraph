# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/mllm.py
# Description: 定义多模态大模型类

from .config import VisualConfig
import torch
from modelscope import AutoModel, AutoTokenizer
from abc import ABC, abstractmethod


class MLLM(ABC):

    def __init__(self, path: str,
                 config: VisualConfig = VisualConfig()) -> None:
        """ 多模态大模型

        Args:
            path (str): 模型名称或路径
            config (VisualConfig, optional): _description_. Defaults to VisualConfig().
        """
        pass

    @abstractmethod
    def chat(self, msgs: list, sys_prompt: str = None) -> str:
        """ 问答任务

        Args:
            msgs (list): 输入内容
            sys_prompt (str, optional): 系统提示词. Defaults to None.

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            str: 模型输出
        """
        raise NotImplementedError


class MiniCPM(MLLM):

    def __init__(self, path: str,
                 config: VisualConfig = VisualConfig()) -> None:
        """ MiniCPM系列模型, 执行图片问答任务

        Args:
            path (str): 模型名称或路径
            config (VisualConfig, optional): 配置. Defaults to VisualConfig().
        """
        super().__init__(path, config)
        self.model = AutoModel.from_pretrained(path,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16)
        self.model = self.model.to(device='cuda')

        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.model.eval()
        self.config = config

    def chat(self, msgs: list, sys_prompt: str = None) -> str:
        """ 图片问答

        Args:
            msgs (list): 输入内容.
            sys_prompt (str, optional): 系统提示词. Defaults to None.

        Returns:
            str: 模型输出
        """

        return self.model.chat(image=None,
                               msgs=msgs,
                               tokenizer=self.tokenizer,
                               sampling=True,
                               temperature=self.config.temperature,
                               sys_prompt=sys_prompt)
