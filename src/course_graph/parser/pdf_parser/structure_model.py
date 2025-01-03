# -*- coding: utf-8 -*-
# Create Date: 2024/10/25
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/structure_model.py
# Description: 布局分析模型封装

from abc import ABC, abstractmethod
from typing_extensions import Required, TypedDict, Literal
from numpy import ndarray
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from paddleocr import PPStructure
from doclayout_yolo import YOLOv10
import json
from course_graph_ext import structure_post_process


class StructureResult(TypedDict, total=False):
    origin_type: Required[Literal[
            'abandon', 'text', 'title', 'figure', 'figure_caption', 'table', 'table_caption', 'table_footnote', 'formula', 'formula_caption'
        ]]
    type: Required[str]
    bbox: Required[tuple[float, float, float, float]]
    text: str


class StructureModel(ABC):

    @abstractmethod
    def predict(self, img: ndarray) -> list[StructureResult]:
        """ 生成布局分析结果

        Args:
            img (ndarray): 图像数组

        Returns:
            list[StructureResult]: 布局分析结果
        """
        raise NotImplementedError

    def __call__(self, img: ndarray) -> list[StructureResult]:
        return self.predict(img)


class PaddleStructure(StructureModel):

    def __init__(self) -> None:
        """ 飞桨布局分析模型 ref: https://github.com/PaddlePaddle/PaddleOCR/
        """
        super().__init__()
        self.pp = PPStructure(table=False, ocr=True, show_log=False)
        self.origin2type = {
            'header': 'abandon',
            'footer': 'abandon',
            'reference': 'abandon',
            'equation': 'formula'
        }

    def predict(self, img: ndarray) -> list[StructureResult]:
        # labels: text, title, figure, figure_caption, table, table_caption, header, footer, reference, equation
        result = self.pp(img)
        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)

        return [{
            'origin_type': item['type'],
            'bbox': tuple(item['bbox']),
            'type': self.origin2type.get(item['type'], item['type'])
        } for item in res]


class LayoutYOLO(StructureModel):

    def __init__(self, model_path: str, device: str = 'cuda', conf: float = 0.2) -> None:
        """ DocLayout-YOLO 布局分析模型 ref: https://github.com/opendatalab/DocLayout-YOLO/blob/main/README-zh_CN.md

        Args:
            model_path (str): 模型路径
            device (str, optional): 运行设备. Defaults to 'cuda'.
            conf (float, optional): 置信度阈值. Defaults to 0.2.
        """
        super().__init__()
        self.model = YOLOv10(model_path)
        self.device = device
        self.origin2type = {
            'plain text': 'text',
            'isolate_formula': 'formula'
        }
        self.conf = conf

    def predict(self, img: ndarray) -> list[StructureResult]:
        result = json.loads(
            self.model.predict(img,
                               imgsz=1024,
                               conf=self.conf,
                               verbose=False,
                               device=self.device)[0].tojson())
        # 将 bbox 坐标变换为 (x1,y1,x2,y2) 格式
        for item in result:
            item['bbox'] = (item['box']['x1'], item['box']['y1'],
                            item['box']['x2'], item['box']['y2'])
        _, w, _ = img.shape
        res = sorted_layout_boxes(result, w)

        # 后处理 (接受元组类型)
        res = structure_post_process(detections=[(item['name'], item['bbox']) for item in res], iou_threshold=0.1)

        return [
            {
                'origin_type': item[0],
                'bbox': item[1],
                'type': self.origin2type.get(item[0], item[0])
            } for item in res
        ]
