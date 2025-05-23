# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/resource.py
# Description: 定义资源类

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pptx import Presentation
from ..llm import LLM
from ..llm.prompt import VLPrompt
from .utils import pptx2imgs
import shutil
from tqdm import tqdm


@dataclass
class Slice:
    """ 资源切片
    """
    file_path: str
    start: int | str
    end: int | str

    def __repr__(self) -> str:
        return f"Slice<path={self.file_path}, range=[{self.start},{self.end}]>"


class Resource(ABC):

    def __init__(self, file_path: str) -> None:
        """ 资源基类

        Args:
            file_path (str): 资源文件路径
        """
        super().__init__()
        self.file_path = file_path

    def __repr__(self) -> str:
        return f"Resource<path={self.file_path}>"

    @abstractmethod
    def get_slices(self, keyword: str) -> list[Slice]:
        """ 通过关键词获取切片
        """
        raise NotImplementedError


def _merge_index_slice(items: list[int], file_path: str) -> list[Slice]:
    """ 将连续的页数索引合并

    Args:
        items (list[int]): 页数列表
        file_path (str): 文件路径

    Returns:
        list[Slice]: 切片列表
    """
    if not items:
        return []

    items.sort()
    slices = []
    start = items[0]
    end = items[0]

    for i in range(1, len(items)):
        if items[i] == end + 1:
            end = items[i]
        else:
            slices.append(Slice(start=start, end=end, file_path=file_path))
            start = items[i]
            end = items[i]

    slices.append(Slice(start=start, end=end, file_path=file_path))
    return slices


class PPTX(Resource):

    def __init__(self, pptx_path: str, vl_prompt: VLPrompt = VLPrompt()) -> None:
        """ .pptx类型文件资源

        Args:
            pptx_path (str): 文件路径
            vl_prompt (VLPrompt, optional): 视觉模型提示词. Defaults to VLPrompt().
        """
        super().__init__(pptx_path)
        self.pptx = Presentation(pptx_path)
        self.vl_prompt = vl_prompt
        # 每一页对应的描述
        self.index_maps: dict[int, str] = dict()

    def __getstate__(self):
        """ 自定义序列化方法
        """
        state = self.__dict__.copy()
        # 移除 parser 属性
        del state['pptx']
        return state

    def __setstate__(self, state):
        """ 自定义反序列化方法
        """
        self.__dict__.update(state)
        self.pptx = Presentation(state['file_path'])

    def get_slices(self, keyword: str) -> list[Slice]:
        """ 通过关键词获取切片

        Args:
            keyword (str): 关键词

        Returns:
            list[Slice]: 切片列表
        """
        # 描述中提到了关键词
        idxs = [key for key, val in self.index_maps.items() if keyword in val]
        return _merge_index_slice(idxs, self.file_path)

    def set_maps_by_vlm(self, model: LLM) -> None:
        """ 使用视觉模型提取pptx主要内容

        Args:
            model (LLM): 视觉模型
        """
        cache_path = '.cache/pptx_imgs_cache'
        imgs = pptx2imgs(self.file_path, cache_path)
        res = ''
        for idx, img in tqdm(enumerate(imgs), total=len(imgs)):
            if idx == 0:
                prompt_, instruction = self.vl_prompt.get_ie_prompt()
                model.instruction = instruction
                res = model.image_chat(img, prompt_)
            else:
                prompt_, instruction = self.vl_prompt.get_context_ie_prompt(res)  # 之前的回答作为上文信息，可以更好理解本张图片
                model.instruction = instruction
                res = model.image_chat([imgs[idx - 1], img], prompt_)
            # 页数从1开始
            self.index_maps[idx + 1] = res
        # 删除缓存文件夹
        shutil.rmtree(cache_path)


@dataclass
class ResourceMap:
    """ 资源和书签的关联关系
    """
    bookmark_title: str
    resource: Resource
