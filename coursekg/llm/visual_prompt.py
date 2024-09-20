# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/visual_prompt.py
# Description: 定义多模态大模型提示词类

from dataclasses import dataclass
from abc import abstractmethod, ABC
from PIL import Image
import os
import json
import random


@dataclass
class Interaction:
    image_paths: str | list[str]
    question: str
    answer: str


class VisualPrompt(ABC):

    def __init__(self) -> None:
        """ 视觉提示词
        """
        self.prompt: str = ''
        self.sys_prompt: str | None = None

    @abstractmethod
    def set_type_ocr(self) -> None:
        """ OCR

        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    def set_type_context_ie(self, message: str) -> None:
        """ 带有上文信息的信息提取

        Args:
            message str: 上文信息.

        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def set_type_ie(self) -> None:
        """ 信息提取

        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def set_type_catalogue(self) -> None:
        """ 目录识别

        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def use_examples(
            self,
            example_dataset_path: str = 'dataset/image_example'
    ) -> 'VisualPrompt':
        """ 添加示例

        Args:
            example_dataset_path (str, optional): 使用多模态模型上下文学习源数据地址文件夹. Defaults to 'dataset/image_example'.
        
        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def add_history(self, history: Interaction) -> 'VisualPrompt':
        """ 输入历史记录以进行多轮问答

        Args:
            history (Interaction): 一条问答记录
        
        Raises:
            NotImplementedError: 子类需要实现该方法
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, image_path: str | list[str]) -> list:
        """ 获取提示词

        Args:
            image_path (str, list[str]): 图片路径
        
        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list: 组装后的提示词
        """
        raise NotImplementedError

    def get_sys_prompt(self) -> str | None:
        """ 获取系统提示词

        Returns:
            (str | None): 系统提示词
        """
        return self.sys_prompt


class MiniCPMPrompt(VisualPrompt):

    def __init__(self) -> None:
        """ MiniCPM提示词, 支持多轮对话, 多图对话和上下文学习
        """
        super().__init__()
        self.interactions: list[Interaction] = []
        self.type_ = ''

    def set_type_ocr(self) -> None:
        """ ocr
        """
        self.prompt = """将图片中识别到的文字转换文本输出。你必须做到：
        1. 你的回答中严禁包含 “以下是根据图片内容生成的文本：”或者 “ 将图片中识别到的文字转换文本格式输出如下：” 等这样的提示语。
        2. 不需要对内容进行解释和总结。
        3. 代码包含在``` ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式。
        4. 如果图片中包含图表，对图表形成摘要即可，无需添加例如“图片中的文本内容如下：”等这样的提示语。
        再次强调，不要输出和识别到的内容无关的文字。"""
        self.sys_prompt = '你是一个OCR模型。'
        self.type_ = 'ocr'

    def set_type_ie(self) -> None:
        """ 信息提取
        """
        self.prompt = '请帮我提取图片中的主要内容'
        self.sys_prompt = '你是一个能够总结图片内容的模型。'
        self.type_ = 'ie'

    def set_type_catalogue(self) -> None:
        """ 判断目录
        """
        self.prompt = """给你一张图片，他是书籍中的某一页。请你帮我判断这一页是不是这本书的目录页中的其中一页。你必须做到：
        1.只需要回答我是或否这一个字即可。不需要其他任何的解释。
        2.书籍的目录页是指包含一些章节名称和对应的页码。书籍的封面、作者介绍、版权页、前言和正文等都不能算作目录页。"""
        self.sys_prompt = '你是一个能够总结图片内容的模型。'
        self.type_ = 'catalogue'

    def set_type_context_ie(self, message: str) -> None:
        """ 带有上文信息的信息提取

        Args:
            message str: 上文信息.
        """
        self.prompt = f'''第一张图片的主要内容是：{message},
        第一张图片和第二张图片在文档中是顺序出现的，请你据此帮我总结第二张图片的主要内容'''
        self.sys_prompt = '你是一个能够总结图片内容的模型。'
        self.type_ = 'context_ie'

    def use_examples(
        self,
        example_dataset_path: str = 'dataset/image_example'
    ) -> 'MiniCPMPrompt':
        """ 添加示例 (示例也可以是上文对话)

        Args:
            example_dataset_path (str, optional): 使用多模态模型上下文学习源数据地址文件夹. Defaults to 'dataset/image_example'.
        """
        self.interactions = []
        with open(os.path.join(example_dataset_path, 'example.json')) as f:
            examples = json.load(f)
            for line in examples:
                if line['type'] == self.type_:
                    self.interactions.append(
                        Interaction(line['image'], self.prompt,
                                    line['output']))
        if len(self.interactions) > 5:
            self.interactions = random.sample(self.interactions, 5)
        return self

    def add_history(self, history: Interaction) -> 'MiniCPMPrompt':
        """ 输入历史记录以进行多轮问答

        Args:
            history (Interaction): 一条问答记录
        """
        self.interactions.append(history)
        return self

    def get_prompt(self, image_path: str | list[str]) -> list:
        """ 获取提示词

        Args:
            image_path (str, list[str]): 图片路径

        Returns:
            list: 组装后的提示词
        """

        def get_content(image_paths, question) -> list:
            if len(image_paths) == 0:
                content = []
            elif isinstance(image_paths, str):
                content = [Image.open(image_paths).convert('RGB')]
            else:
                content = [
                    Image.open(path).convert('RGB') for path in image_paths
                ]
            content.append(question)
            return content

        msgs = []
        for example in self.interactions:
            msgs.append({
                'role':
                'user',
                'content':
                get_content(example.image_paths, example.question)
            })
            msgs.append({'role': 'assistant', 'content': [example.answer]})

        msgs.append({
            'role': 'user',
            'content': get_content(image_path, self.prompt)
        })
        return msgs
