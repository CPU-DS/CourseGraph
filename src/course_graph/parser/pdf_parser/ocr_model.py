# -*- coding: utf-8 -*-
# Create Date: 2024/10/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/ocr_model.py
# Description: OCR 模型封装

from abc import ABC, abstractmethod
from paddleocr import PaddleOCR as Paddle
from modelscope import AutoModel, AutoTokenizer
import logging
from contextlib import redirect_stdout
import os
import re
import time
from ...llm import Qwen

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
        res = self.paddle.ocr(img_path)[0]
        if res is not None:
            sts = [line[1][0] for line in res]
            return re.sub(r'[^\S\n]+', '', ''.join(sts))
        return ''
        

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
        
        self.unreadable_pattern = re.compile(r'[\ue000-\uf8ff\ufff0-\uffff]')

    class OverrideGenerate:
        def __init__(self, model, temperature: float = 1.0, do_sample: bool = True):
            self.model = model
            self.original_generate = model.generate
            self.temperature = temperature
            self.do_sample = do_sample

        def __enter__(self):
            def new_generate(*args, **kwargs):
                kwargs['temperature'] = self.temperature
                kwargs['do_sample'] = self.do_sample
                return self.original_generate(*args, **kwargs)
            self.model.generate = new_generate

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.model.generate = self.original_generate

    def predict(self, img_path: str) -> str:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
            timmout = False 
            start_time = time.time()
            res = self.model.chat(self.tokenizer, img_path, ocr_type='ocr')
            if time.time() - start_time > 30:
                timmout = True
            res = res.replace('\n', '').replace('\u3000', ' ')
            
            if self.unreadable_pattern.search(res) or timmout:
                retry = 0
                while retry < 3: # 不断尝试提高温度
                    with self.OverrideGenerate(self.model, temperature=1.0 * (retry + 1), do_sample=True):
                        res = self.model.chat(self.tokenizer, img_path, ocr_type='ocr')
                    if not self.unreadable_pattern.search(res):
                        break
                    retry += 1
                else:
                    res = self.unreadable_pattern.sub('', res)  # 过滤掉不可读的文本
            return res

class QwenOCR(OCRModel):

    def __init__(self, api_key: str = os.getenv("DASHSCOPE_API_KEY")) -> None:
        """ qwen-vl-ocr 模型 ref: https://help.aliyun.com/zh/model-studio/user-guide/qwen-vl-ocr
        
        Args:
            api_key (str, optional): API key. Defaults to os.getenv("DASHSCOPE_API_KEY").
        """
        self.model = Qwen(name='qwen-vl-ocr', api_key=api_key)

    def predict(self, img_path: str) -> str:
        return self.model.image_chat([img_path], 'Read all the text in the image.')
