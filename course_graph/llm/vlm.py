# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/vlm.py
# Description: 定义图文理解模型类

from .config import vlm_config
import torch
from modelscope import AutoModel, AutoTokenizer
from PIL import Image

def get_msgs(image_paths, message) -> list:
    if len(image_paths) == 0:
        msgs = []
    elif isinstance(image_paths, str):
        msgs = [Image.open(image_paths).convert('RGB')]
    else:
        msgs = [Image.open(path).convert('RGB') for path in image_paths]
    msgs.append(message)
    return msgs



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

    def chat(self, image_paths: str | list[str], message: str) -> str:
        """ 图片问答

        Args:
            image_paths (str | list[str]): 多张图片
            message (str): 用户输入


        Returns:
            str: 模型输出
        """
        return self.model.chat(image=None,
                               msgs=get_msgs(image_paths, message),
                               tokenizer=self.tokenizer,
                               sampling=True,
                               temperature=vlm_config.temperature,
                               sys_prompt=self.instruction)
