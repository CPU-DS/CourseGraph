# -*- coding: utf-8 -*-
# Create Date: 2024/10/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/ocr_model.py
# Description: OCR 模型封装

from abc import ABC, abstractmethod
from paddleocr import PaddleOCR as Paddle
from modelscope import AutoModel, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)


class OCRModel(ABC):

    @abstractmethod
    def predict(self, img_path: str) -> str:
        """ OCR 识别

        Args:
            img_path (str): 图像路径

        Returns:
            str: 识别结果
        """
        raise NotImplementedError

    def __call__(self, img_path) -> str:
        return self.predict(img_path)


class PaddleOCR(OCRModel):

    def __init__(self) -> None:
        """ 飞桨 OCR 模型 ref: https://github.com/PaddlePaddle/PaddleOCR/
        """
        self.paddle = Paddle(lang="ch", show_log=False, use_angle_cls=True)

    def predict(self, img_path: str) -> str:
        sts = []
        for line in self.paddle.ocr(img_path)[0]:
            sts.append(line[1][0])
        return '\n'.join(sts)


class GOT(OCRModel):

    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        """ GOT-OCR 2.0 模型 ref: https://github.com/Ucas-HaoranWei/GOT-OCR2.0
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id).eval().to(device)

    def predict(self, img_path: str) -> str:
        return self.model.chat(self.tokenizer, img_path, ocr_type='ocr')
