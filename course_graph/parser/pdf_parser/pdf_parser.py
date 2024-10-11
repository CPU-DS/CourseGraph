# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/pdf_parser.py
# Description: 定义pdf文档解析器

from ..base import BookMark
from .pdf_parser_mode import *
import uuid
from ..parser import Parser, Page, Content, ContentType
import fitz
from paddleocr import PPStructure
from PIL import Image
import numpy as np
import cv2
import re
from ...llm import MLM, VLUPrompt, LLM, ParserPrompt, Model
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
import os
import shutil
import ast
from modelscope import AutoModel, AutoTokenizer


def _replace_linefeed(sentence: str, ignore_end=True, replace='') -> str:
    """ 移除句子的换行符

    Args:
        sentence (str): 句子
        ignore_end (bool, optional): 忽略句末的换行符. Defaults to True.
        replace (str, optional): 换行符替换对象. Defaults to ''.

    Returns:
        str: 新句
    """
    if ignore_end:
        sentence_endings = r'[。！？.!?]'
        pattern = r'(?<!' + sentence_endings + r')\n'
    else:
        pattern = r'\n'
    sentence = re.sub(pattern, replace, sentence)
    return sentence


def get_list(text: str) -> list:
    """ 括号匹配找到列表

    Args:
        text (str): 文本

    Returns:
        list: 列表
    """
    list_string = ''
    stack = 0
    for s in text:
        if s == '[':
            stack += 1
        if stack > 0:
            list_string += s
        if s == ']':
            stack -= 1
            if stack == 0:
                break
    return ast.literal_eval(list_string)


class PDFParser(Parser):

    def __init__(self, pdf_path: str) -> None:
        """ 解析pdf文档, 需要带有书签以判断层级

        Args:
            pdf_path (str): pdf文档路径
        """
        super().__init__(pdf_path)
        self._pdf = fitz.open(pdf_path)

        self.parser_mode: PDFParserMode = BaseMode()

        self._pp = PPStructure(table=False, ocr=True, show_log=False)
        self.outline: list = outline if len(
            outline := self._pdf.get_toc()) != 0 else []
        # 可以使用 simple=False 获得更详细的信息，包含锚点等
        # 这里如果不自带书签则需要手动制定目录页，读取文字再使用大模型解析
        self._got = None

    def set_ocr_engine_got(self,
                           model_path: str = 'stepfun-ai/GOT-OCR2_0'
                           ) -> 'PDFParser':
        """ 使用GOT模型作为OCR引擎, 默认使用paddle OCR
            reference: https://github.com/Ucas-HaoranWei/GOT-OCR2.0

        Args:
            model_path (str, optional): 模型名称或路径. Defaults to 'stepfun-ai/GOT-OCR2_0'.

        Returns:
            PDFParser: PDFParser
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        self._got = Model(model=AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=tokenizer.eos_token_id).eval().cuda(),
                          tokenizer=tokenizer)
        return self

    def __enter__(self) -> 'PDFParser':
        return self

    def close(self) -> None:
        """ 关闭文档
        """
        self._pdf.close()

    def get_catalogue_index_by_visual_model(
            self,
            visual_model: MLM,
            visual_prompt: VLUPrompt,
            rate: float = 0.1) -> tuple[int, int]:
        """ 通过多模态大模型寻找目录页, 返回目录页起始页和终止页页码 (从0开始编序)

        Args:
            visual_model (MLM): 多模态大模型
            visual_prompt (VLUPrompt): 图文理解提示词
            rate (float, optional): 查询前 ratio 比例的页面. Defaults to 0.1 即 10%.

        Returns:
            tuple[int, int]: 目录页起始页和终止页页码
        """
        visual_prompt.set_type_catalogue()
        cache_path = '.cache/pdf_cache'
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        catalogue = []
        for index in range(int(self._pdf.page_count * rate)):
            img = self._get_page_img(index, zoom=2)
            file_path = os.path.join(cache_path, f'{index}.png')
            Image.fromarray(img).save(file_path)
            res = visual_model.chat(msgs=visual_prompt.get_prompt(file_path),
                                    sys_prompt=visual_prompt.get_sys_prompt())
            if res == '是' or res == '是。' or res == '是.':
                catalogue.append(index)
        shutil.rmtree(cache_path)

        def find_longest_consecutive_sequence(
                nums: list[int]) -> tuple[int, int]:
            """ 找到一个最长的连续序列的起点和终点

            Args:
                nums (list[int]): 序列

            Returns:
                tuple[int, int]: 起点数字和终点数字
            """
            if not nums:
                return -1, -1

            nums = sorted(set(nums))
            max_start = max_end = nums[0]
            current_start = nums[0]
            max_length = 1
            current_length = 1

            for i in range(1, len(nums)):
                if nums[i] == nums[i - 1] + 1:
                    current_length += 1
                    if current_length > max_length:
                        max_length = current_length
                        max_start = current_start
                        max_end = nums[i]
                else:
                    current_start = nums[i]
                    current_length = 1

            return max_start, max_end

        return find_longest_consecutive_sequence(catalogue)

    def _set_outline(self,
                     lines: list,
                     offset: int,
                     llm: LLM,
                     parser_prompt: ParserPrompt = ParserPrompt) -> None:
        """ 设置大纲层级

        Args:
            lines (list): 标题和页码数组
            offset (int): 页码偏移
            llm (LLM): 大模型
            parser_prompt (ParserPrompt, optional):  文档解析提示词类. Defaults to ParserPrompt.
        """
        lines_without_index = [line[0] for line in lines]
        res = llm.chat(parser_prompt.get_outline_prompt(lines_without_index))
        r2 = get_list(res)

        outline = []
        for i in range(len(r2)):
            level = int(re.findall(r'\d+', r2[i][1])[0])
            if str(lines[i][1]).isdigit():
                outline.append([level, lines[i][0], int(lines[i][1]) + offset])
        self.outline = outline

    def set_outline_by_catalogue(
            self,
            start_index: int,
            end_index: int,
            offset: int,
            llm: LLM,
            parser_prompt: ParserPrompt = ParserPrompt) -> None:
        """ 手动指定目录页, 通过大模型解析目录页获取大纲层级

        Args:
            start_index (int): 目录页起始页 (从0开始编序)
            end_index (int): 目录页终止页
            offset (int): 首页偏移
            llm (LLM): 大模型
            parser_prompt (ParserPrompt, optional): 文档解析提示词类. Defaults to ParserPrompt.
        """

        self.outline = []
        page_index = list(range(start_index, end_index + 1))
        lines = []
        for index in page_index:
            page = self._pdf[index]
            text = page.get_text()
            if len(text) == 0:  # 图片型pdf则使用OCR
                text = '\n'.join([
                    item['text'] for item in self._page_structure(
                        self._get_page_img(index, zoom=2))
                ])
            res = llm.chat(parser_prompt.get_directory_prompt(text)).replace(
                "，", ",")
            lines.extend(get_list(res))
        self._set_outline(lines, offset, llm, parser_prompt)

    def set_outline_auto(self,
                         llm: LLM,
                         parser_prompt: ParserPrompt = ParserPrompt) -> None:
        """ 使用OCR自动设置大纲层级, 适用于没有目录页的情况

        Args:
            llm (LLM): 大模型
            parser_prompt (ParserPrompt, optional): 文件解析提示词类. Defaults to ParserPrompt.
        """
        titles = []
        for index in range(self._pdf.page_count):
            img = self._get_page_img(index, zoom=1)
            res = self._page_structure(img)
            titles.extend([[block['text'], index] for block in res
                           if block['type'] == 'title'])
        self._set_outline(titles, 0, llm, parser_prompt)

    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Returns:
            list[BookMark]: 书签列表
        """
        stack: list[BookMark] = []
        bookmarks: list[BookMark] = []
        if len(self.outline) == 0:
            raise ValueError('请先通过 set_outline 方法手动设置大纲')
        for item in self.outline:
            level, title, page = item
            page -= 1  # 从0开始
            level -= 1  # 从0开始
            bookmarks.append(
                BookMark(
                    id='1:' + str(uuid.uuid4()) + f':{level}',
                    title=title,
                    page_index=page,
                    page_end=0,  # 结束页码需要由下一个书签确定
                    level=level,
                    subs=[],
                    resource=[]))

        for bookmark in reversed(bookmarks):
            level = bookmark.level

            while stack and stack[-1].level > level:
                bookmark.subs.append(stack.pop())

            stack.append(bookmark)

        stack.reverse()

        # 设置各个书签的结束页码
        def set_page_end(bks: list[BookMark]):
            for idx in range(len(bks)):
                if idx != len(bks) - 1:
                    bks[idx].set_page_end(bks[idx + 1].page_index)
                set_page_end(bks[idx].subs)

        set_page_end(stack)
        stack[-1].set_page_end(self._pdf.page_count - 1)

        return stack

    def get_content(self, bookmark: BookMark) -> list[Content]:
        """  获取书签下的所有内容

        Args:
            bookmark (BookMark): 书签

        Returns:
            list[Content]: 内容列表
        """
        # 获取书签对应的页面内容
        contents: list[Content] = []
        # 后续这个地方可以并行执行
        for index in range(bookmark.page_index, bookmark.page_end + 1):

            # 起始页内容定位
            page_contents = self.get_page(index).contents
            if index == bookmark.page_index:
                idx = 0
                for i, content in enumerate(page_contents):
                    blank_pattern = re.compile(r'\s+')  # 可能会包含一些空白字符这里去掉
                    content_new = re.sub(blank_pattern, '', content.content)
                    title_new = re.sub(blank_pattern, '', bookmark.title)
                    if content.type == ContentType.Title and (
                            content_new == title_new
                            or content_new in title_new):
                        idx = i + 1
                        break
                page_contents = page_contents[idx:]
            # 终止页内容定位
            if index == bookmark.page_end:
                idx = len(page_contents)
                for i, content in enumerate(page_contents):
                    if content.type == ContentType.Title:  # 直到遇到下一个标题为止，这里的逻辑可能存在问题~
                        idx = i
                        break
                page_contents = page_contents[:idx]
            contents.extend(page_contents)
        return contents

    def _get_page_img(self, page_index: int, zoom: int = 1):
        """ 获取页面的图像对象

        Args:
            page_index (int): 页码
            zoom (int, optional): 缩放倍数. Defaults to 1.

        Returns:
            _type_: opencv 转换后的图像对象
        """
        pdf_page = self._pdf[page_index]
        # 不需要对页面进行缩放
        mat = fitz.Matrix(zoom, zoom)
        pm = pdf_page.get_pixmap(matrix=mat, alpha=False)
        # 图片过大则放弃缩放
        if pm.width > 2000 or pm.height > 2000:
            pm = pdf_page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def _page_structure(self, img) -> list[dict]:
        """ 使用PP-Structure进行版面分析

        Args:
            img (_type_): 图像对象

        Returns:
            list[dict]: 识别后的结果
        """
        result = self._pp(img)
        h, w, _ = img.shape
        res = sorted_layout_boxes(result, w)
        return [
            {
                'type': item['type'],  # 区域类型
                'bbox': item['bbox'],  # 区域边界坐标
                'text': ''.join([r['text'] for r in item['res']])  # 文字OCR结果
            } for item in res
        ]

    def get_page(self, page_index: int) -> Page:
        """ 获取文档页面

        Args:
            page_index (int): 页码, 从0开始计数

        Returns:
            Page: 文档页面
        """
        match self.parser_mode:
            case PaddleMode():
                pdf_page = self._pdf[page_index]
                img = self._get_page_img(page_index, zoom=2)
                blocks = self._page_structure(img)
                contents: list[Content] = []
                for block in blocks:
                    content = pdf_page.get_textbox(block['bbox'])
                    if block['type'] == 'text':
                        content = _replace_linefeed(content)
                        contents.append(
                            Content(type=ContentType.Text, content=content))
                    elif block['type'] == 'title':
                        contents.append(
                            Content(type=ContentType.Title, content=content))
            case BaseMode():
                pdf_page = self._pdf[page_index]
                contents = [
                    Content(type=ContentType.Text, content=pdf_page.get_text())
                ]
            case _:
                zoom = 2
                pdf_page = self._pdf[page_index]
                img = self._get_page_img(page_index, zoom=zoom)
                h, w, _ = img.shape
                blocks = self._page_structure(img)

                t = 20
                # 切割子图, 向外扩充t个像素
                cache_path = '.cache/pdf_cache'
                if not os.path.exists(cache_path):
                    os.mkdir(cache_path)
                contents: list[Content] = []

                for idx, block in enumerate(blocks):
                    type_ = block['type']
                    if type_ in ['header', 'footer', 'reference']:
                        continue  # 页眉页脚注释部分不要
                    x1, y1, x2, y2 = block['bbox']
                    # 扩充裁剪区域
                    x1, y1, x2, y2 = max(0, x1 - t), max(0, y1 - t), min(
                        w, x2 + t), min(h, y2 + t)  # 防止越界
                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        continue  # 区域过小
                    if type_ == 'figure' and ((x2 - x1) < 150 or
                                              (y2 - y1) < 150):
                        continue  # 图片过小
                    cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
                    file_path = os.path.join(cache_path, f'{idx}_{type_}.png')
                    cropped_img.save(file_path)

                    match self.parser_mode:
                        case VisualMode(model, prompt):
                            model: MLM
                            prompt: VLUPrompt
                            res = model.chat(
                                msgs=prompt.get_prompt(file_path),
                                sys_prompt=prompt.get_sys_prompt())

                        case CombinationMode(visual_model, visual_prompt, llm):
                            visual_model: MLM
                            visual_prompt: VLUPrompt
                            llm: LLM | None
                            if type_ in ['title', 'text']:  # 直接读取或OCR + 大模型矫正
                                bbox = [b / zoom for b in block['bbox']]
                                res = pdf_page.get_textbox(bbox).replace(
                                    '\n', '')  # 直接读取可能存在不正确地换行
                                # 有些pdf是图片型可能无法直接读取, 则使用OCR的结果
                                if len(res) == 0:
                                    res = block['text']
                                if llm is not None:
                                    try:
                                        res = llm.chat(
                                            ParserPrompt.get_ocr_aided_prompt(
                                                res))
                                    finally:
                                        pass  # 这一步不是必须的
                            else:
                                res = visual_model.chat(
                                    msgs=visual_prompt.get_prompt(file_path),
                                    sys_prompt=visual_prompt.get_sys_prompt())
                        case _:
                            res = ''
                    if block['type'] == 'title':
                        contents.append(
                            Content(type=ContentType.Title, content=res))
                    else:  # 其余全部当作正文对待
                        res = _replace_linefeed(res)
                        contents.append(
                            Content(type=ContentType.Text, content=res))
                shutil.rmtree(cache_path)
        return Page(page_index=page_index + 1, contents=contents)

    def get_pages(self) -> list[Page]:
        """ 获取pdf文档所有页面

        Returns:
            list[Page]: 页面列表
        """
        pages: list[Page] = []
        for index in range(0, self._pdf.page_count):
            pages.append(self.get_page(page_index=index))
        return pages
