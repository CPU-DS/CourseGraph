# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/pdf_parser/pdf_parser.py
# Description: 定义pdf文档解析器

from .structure_model import *
from .ocr_model import *
import shortuuid
from ..parser import Parser
from ..type import Page, Content, ContentType
import fitz
from PIL import Image
import numpy as np
import cv2
import re
from ...llm import VLM, LLM
from ...llm.prompt import VLPromptGenerator, ParserPromptGenerator
import os
import shutil
from course_graph_ext import get_list_from_string, find_longest_consecutive_sequence
from ..type import BookMark, PageIndex


class PDFParser(Parser):

    def __init__(
            self,
            pdf_path: str,
            ocr_model: OCRModel = PaddleOCR(),
            ocr_priority: bool = False,
            structure_model: StructureModel = PaddleStructure(),
            parser_prompt: ParserPromptGenerator = ParserPromptGenerator(),
            vl_prompt: VLPromptGenerator = VLPromptGenerator(),
            vlm: VLM = None,
            llm: LLM = None,
            anchor: bool = False,
            sharpen: Literal['USM', 'Laplacian'] | None = None,
            **kwargs
    ) -> None:
        """ pdf文档解析器

        Args:
            pdf_path (str): pdf文档路径
            ocr_model (OCRModel, optional): OCR 模型. Defaults to PaddleOCR().
            ocr_priority (bool, optional): 获取文字内容时优先使用 OCR模型, 否则优先直接读取. Defaults to False.
            structure_model (StructureModel, optional): 布局分析模型. Defaults to Paddle().
            parser_prompt (ParserPromptGenerator, optional): 解析提示词. Defaults to ParserPromptGenerator().
            vl_prompt (VLPromptGenerator, optional): 图文理解模型提示词. Defaults to VLPromptGenerator().
            vlm ( VLM, optional): 视觉模型. Default to None.
            llm ( LLM, optional): 语言模型. Default to None.
            anchor (bool, optional): 优先使用锚点定位. Defaults to False.
            sharpen (Literal['USM', 'Laplacian'] | None, optional): 锐化处理算法. Defaults to None.
            **kwargs (dict, optional): 其它细粒度控制参数.
        """
        super().__init__(pdf_path)
        self._pdf = fitz.open(pdf_path)

        self.structure_model = structure_model
        self.ocr_model = ocr_model
        self.ocr_priority = ocr_priority

        self.parser_prompt = parser_prompt
        self.vl_prompt = vl_prompt
        self.vlm = vlm
        self.llm = llm

        self.anchor = anchor
        self.sharpen = sharpen
        
        self.kwargs = kwargs

        self.outline: list[list] = self._get_outline()

    def _get_outline(self) -> list[list]:
        """ 从 pdf 中读取大纲层级

        Returns:
            list[list]: 大纲层级
        """
        outline = []
        for item in self._pdf.get_toc(simple=False):
            h = self._pdf[item[2] - 1].get_pixmap().height  # 宽高一律采用像素层面
            if self.anchor:
                match item[3]['kind']:
                    case 4:
                        try:
                            xref = item[3]['xref']
                            t_xref = int(self._pdf.xref_get_key(xref, 'A')[1].split()[0])
                            fitH = int(self._pdf.xref_get_key(t_xref, 'D')[1][1:-1].split()[-1])
                            outline.append([*item[:3], (-1, max(h - fitH, 0))])
                        except:
                            outline.append([*item[:3], (-1, -1)])  # 解析出错
                    case 1:
                        outline.append([*item[:3], (item[3]['to'].x, item[3]['to'].y)])
                    case _:
                        outline.append([*item[:3], (-1, -1)])
            else:
                outline.append([*item[:3], (-1, -1)])
        return outline

    def __enter__(self) -> 'PDFParser':
        return self

    def close(self) -> None:
        """ 关闭文档
        """
        self._pdf.close()

    def get_catalogue_index_by_vlm(
            self,
            vlm: VLM,
            rate: float = 0.1) -> tuple[int, int]:
        """ 通过图文理解模型寻找目录页, 返回目录页起始页和终止页页码 (从0开始编序)

        Args:
            vlm (VLM): 图文理解模型
            rate (float, optional): 查询前 ratio 比例的页面. Defaults to 0.1 即 10%.

        Returns:
            tuple[int, int]: 目录页起始页和终止页页码
        """
        cache_path = '.cache/pdf_cache'
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        catalogue = []
        for index in range(int(self._pdf.page_count * rate)):
            img = self._get_page_img(index, zoom=2)
            file_path = os.path.join(cache_path, f'{index}.png')
            Image.fromarray(img).save(file_path)
            prompt_, instruction = self.vl_prompt.get_catalogue_prompt()
            vlm.instruction = instruction
            res = vlm.chat(file_path, prompt_)
            if res.startswith('是'):
                catalogue.append(index)
        shutil.rmtree(cache_path)

        return find_longest_consecutive_sequence(catalogue)

    def _set_outline(self,
                     lines: list,
                     offset: int,
                     llm: LLM) -> None:
        """ 设置大纲层级

        Args:
            lines (list): 标题和页码数组
            offset (int): 页码偏移
            llm (LLM): 大模型
        """
        lines_without_index = [line[0] for line in lines]
        prompt, instruction = self.parser_prompt.get_outline_prompt(lines_without_index)
        llm.instruction = instruction
        res = llm.chat(prompt)
        r2 = get_list_from_string(res)

        outline: list = []
        for i in range(len(r2)):
            level = int(re.findall(r'\d+', r2[i][1])[0])
            if str(lines[i][1]).isdigit():
                outline.append([level, lines[i][0], int(lines[i][1]) + offset, (-1, -1)])
        self.outline = outline

    def set_outline_by_catalogue(
            self,
            start_index: int,
            end_index: int,
            offset: int,
            llm: LLM) -> None:
        """ 手动指定目录页, 通过大模型解析目录页获取大纲层级

        Args:
            start_index (int): 目录页起始页 (从0开始编序)
            end_index (int): 目录页终止页 (包含终止页)
            offset (int): 首页偏移
            llm (LLM): 大模型
        """

        self.outline = []
        lines = []
        for index in range(start_index, end_index + 1):
            page = self.get_page(index)
            text_contents = '\n'.join(
                [content.content for content in page.contents]).strip()
            prompt, instruction = self.parser_prompt.get_directory_prompt(text_contents)
            llm.instruction = instruction
            res = llm.chat(prompt).replace("，", ",")
            lines.extend(get_list_from_string(res))
        self._set_outline(lines, offset, llm)

    def set_outline_auto(self,
                         llm: LLM) -> None:
        """ 使用OCR自动设置大纲层级, 适用于没有目录页的情况

        Args:
            llm (LLM): 大模型
        """
        titles = []
        for index in range(self._pdf.page_count):
            img = self._get_page_img(index, zoom=1)
            res = self.structure_model(img)
            titles.extend([[block['text'], index] for block in res if block['type'] == 'title'])
        self._set_outline(titles, 0, llm)

    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Returns:
            list[BookMark]: 书签列表
        """
        stack: list[BookMark] = []
        bookmarks: list[BookMark] = []
        for item in self.outline:
            level, title, page, anchor = item
            page -= 1  # 从0开始
            level -= 1  # 从0开始
            bookmarks.append(
                BookMark(id='1:' + str(shortuuid.uuid()) + f':{level}',
                         title=title,
                         page_start=PageIndex(index=page, anchor=anchor),
                         page_end=PageIndex(index=0, anchor=(0, 0)),
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
                    bks[idx].set_page_end(bks[idx + 1].page_start)
                set_page_end(bks[idx].subs)

        set_page_end(stack)
        stack[-1].set_page_end(
            PageIndex(index=self._pdf.page_count - 1, anchor=(-1, -1)))

        return stack

    def get_contents(self, bookmark: BookMark) -> list[Content]:
        """  获取书签下的所有内容

        Args:
            bookmark (BookMark): 书签

        Returns:
            list[Content]: 内容列表
        """

        def remove_blanks(s):
            return re.sub(re.compile(r'\s+'), '', s)

        # 获取书签对应的页面内容
        contents: list[Content] = []
        # 后续这个地方可以并行执行
        for index in range(bookmark.page_start.index, bookmark.page_end.index + 1):

            page_contents = self.get_page(index).contents

            if index == bookmark.page_start.index:
                idx = 0
                x, y = bookmark.page_start.anchor
                title = remove_blanks(bookmark.title)

                if x == -1 and y == -1:  # 使用内容定位
                    condition = lambda s: (
                            s.type == ContentType.Title and remove_blanks(s.content) in remove_blanks(title))
                else:  # 使用 anchor 定位
                    condition = lambda s: (s.bbox[0] > x and s.bbox[1] > y)

                for i, content in enumerate(page_contents):
                    if condition(content):
                        idx = i
                        break

                page_contents = page_contents[idx:]

            if index == bookmark.page_end.index:
                idx = len(page_contents)
                x, y = bookmark.page_start.anchor

                if x == -1 and y == -1:  # 使用内容定位
                    condition = lambda s: (s.type == ContentType.Title)
                else:  # 使用 anchor 定位
                    condition = lambda s: (s.bbox[0] > x and s.bbox[1] > y)
                for i, content in enumerate(page_contents):
                    if condition(content):
                        idx = i
                        break
                page_contents = page_contents[:idx]

            contents.extend(page_contents)
        return contents

    def _get_page_img(self, page_index: int, zoom: int = 1) -> ndarray:
        """ 获取页面的图像对象

        Args:
            page_index (int): 页码
            zoom (int, optional): 缩放倍数. Defaults to 1.

        Returns:
            _type_: opencv 转换后的图像对象
        """
        pdf_page = self._pdf[page_index]
        mat = fitz.Matrix(zoom, zoom)
        pm = pdf_page.get_pixmap(matrix=mat, alpha=False)
        # 图片过大则放弃缩放
        if pm.width > 2000 or pm.height > 2000:
            pm = pdf_page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def get_page(self, page_index: int) -> Page:
        """ 获取文档页面

        Args:
            page_index (int): 页码, 从0开始计数

        Returns:
            Page: 文档页面
        """
        zoom = self.kwargs.get('zoom', 2)
        pdf_page = self._pdf[page_index]
        img = self._get_page_img(page_index, zoom=zoom)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w, _ = img.shape
        
        match self.sharpen:
            case 'USM':  # 非锐化掩膜
                strength = 1
                img_gray = cv2.addWeighted(img_gray, 1 + strength, cv2.GaussianBlur(img_gray, (5, 5), 1.0), -strength, 0) # 加权
            case 'Laplacian':  # 拉普拉斯锐化 
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 改进的拉普拉斯算子
                img_gray = cv2.filter2D(img_gray, -1, kernel)
            case _:
                pass

        # 处理后转回 BGR 格式 (只是为了OCR模型和布局分析模型能够正常使用, 但颜色信息已经丢失了)
        img_sharpen = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        # 文本区域 (text/title) 和布局检测使用 img_sharpen 对象
        # 非文本区域使用 img 对象
        blocks = self.structure_model(img_sharpen)

        cache_path = '.cache/pdf_cache'
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        def save_block(block_: StructureResult, img_: ndarray, idx: int) -> None | str:
            wt, ht = self.kwargs.get('wt', 20), self.kwargs.get('ht', 5)  # 切割子图, 向左右扩充wt, 向上扩充ht
            x1, y1, x2, y2 = block_['bbox']
            type_ = block_['type']
            # 扩充裁剪区域
            x1, y1, x2, y2 = max(0, x1 - wt), max(0, y1 - ht), min(w, x2 + wt), min(h, y2 + ht)  # 防止越界
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                return  # 区域过小
            if type_ == 'figure' and ((x2 - x1) < 150 or (y2 - y1) < 150):
                return  # 图片过小
            # 裁剪图像
            cropped_img = Image.fromarray(img_).crop((x1, y1, x2, y2))
            border_size = self.kwargs.get('cropped_border_size', 20)
            new_size = (cropped_img.width + 2*border_size, cropped_img.height + 2*border_size)
            bordered_img = Image.new('RGB', new_size, 'white')
            bordered_img.paste(cropped_img, (border_size, border_size))
            
            path = os.path.join(cache_path, f'{idx}_{str(shortuuid.uuid())}_{type_}.png')
            bordered_img.save(path)
            return path

        def set_text_auto(block_: StructureResult, idx: int) -> None:
            bbox = [b / zoom for b in block_['bbox']]

            if block_.get('text', None) is not None:  # 已经设置过 text 属性
                pass
            else:
                res = pdf_page.get_textbox(bbox).replace('\n', '')  # 直接读取
                if len(res) != 0 and not bool(re.search(r'[\uFFFD]', res)) and not self.ocr_priority:
                    block_['text'] = res
                elif path := save_block(block_, img_sharpen, idx):  # OCR
                    res = self.ocr_model(path)
                    if self.llm is not None:
                        try:
                            prompt_, instruction_ = self.parser_prompt.get_ocr_aided_prompt(res)
                            self.llm.instruction = instruction_
                            res = self.llm.chat(prompt_)
                        finally:
                            pass  # 使用大模型矫正这一步不是必须的
                    block_['text'] = res
        
        def set_text_by_vlm(block_: StructureResult, idx: int) -> None:
            if file_path := save_block(block_, img, idx):
                prompt, instruction = self.vl_prompt.get_ocr_prompt()
                self.vlm.instruction = instruction
                block_['text'] = self.vlm.chat(file_path, prompt)

        for idx, block in enumerate(blocks):
            type_ = block['type']
            if type_ in ['abandon']:
                continue
            elif type_ in ['title', 'text']:
                set_text_auto(block, idx)  # 直接读取 或 OCR
            else:
                if self.vlm is not None:
                    set_text_by_vlm(block, idx)  # 使用多模态模型

        contents: list[Content] = []
        for block in blocks:
            if content := block.get('text', None):  # 空字符串或None
                content = Content(
                    type=ContentType.Text,
                    content=content,
                    bbox=tuple([b / zoom for b in block['bbox']]),  # 还原为原始大小坐标
                    origin_type=block['origin_type'])
                if block['type'] == 'title':
                    content.type = ContentType.Title  # 除了title其余全部当作正文对待
                contents.append(content)

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
