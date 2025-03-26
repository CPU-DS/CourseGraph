# -*- coding: utf-8 -*-
# Create Date: 2024/11/05
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt/parser_prompt.py
# Description: 使用大模型解析文档相关提示词

class ParserPromptGenerator:

    @staticmethod
    def get_ocr_aided_prompt(text: str) -> tuple[str, str]:
        """ 使用大模型纠正OCR识别中的错误, 获取相应提示词

        Args:
            text (str): 原始识别结果

        Returns:
            str: prompt输出
        """
        return f"""你的任务是输出经过纠正后的准确文字。遵循以下准则:
    1.纠正OCR导致的错别字和错误: 使用上下文和常识来纠正常见的错误，只修复明显的错误，不要不必要地更改内容；
    2.保持原有结构：不要添加额外的句号或任何不必要的标点符号；
    3.保留原始内容：删除句子或段落中不必要的换行符，确保内容与之前的上下文顺畅连接。
    重要提示：仅回复更正后的文本。保留所有原始格式，包括换行符。不要包含任何介绍、解释或元数据。
    OCR结果为: {text}""", '你擅长帮助用户纠正从图片中提取的OCR文字错误。'

    @staticmethod
    def get_directory_prompt(content: str) -> tuple[str, str]:
        """ 使用大模型纠正整理目录, 获取相应提示词

        Args:
            content (str): 原始文字

        Returns:
            str: prompt输出
        """
        return f"""我将给你一段自动获取到的目录文字，请你帮我整理成格式化的形式，要求如下:
    1.每一条目录记录使用列表 [标题, 页码] 的格式进行组织，注意不要去掉标题中包含的序号;
    2.多条目录记录使用列表进行组织，最终返回的应该是一个二维列表;
    3.不要为这个二维列表添加表头;
    示例: 一个包含两条目录记录
    "第6章 与学习相关的技巧· ·····································163
    6.1 参数的更新············································163"
    被整理成如下格式：[
        ["第6章 与学习相关的技巧", 163],
        ["6.1 参数的更新", 163]
    ];
    原文如下：{content}""", '你擅长进行格式的整理。'

    @staticmethod
    def get_outline_prompt(directory_list: list) -> tuple[str, str]:
        """ 使用大模型为目录添加层级

        Args:
            directory_list (list): 目录列表

        Returns:
            str: prompt输出
        """
        prompt = """输入：
    [
        ['第3章 数据获取'],
        ['3.1 前提假设与数据方案设计'],
        ['3.1.1 前提假设'],
        ['3.1.2 数据方案设计'],
        ['3.1.3 数据获取的可行性分析'],
        ['3.1.4 确定数据构成'],
        ['3.2 总体和抽样'],
        ['3.2.1 总体和个体'],
        [3.2.2 样本],
        ['第4章 Python基础'],
        ['4.1 Python的下载与安装']
        ['4.2 常用工具包的下载与安装]
        ['思考题']
    ]
    输出:
    [
        ['第3章 数据获取', '1级标题'],
        ['3.1 前提假设与数据方案设计', '2级标题'],
        ['3.1.1 前提假设', '3级标题'],
        ['3.1.2 数据方案设计', '3级标题'],
        ['3.1.3 数据获取的可行性分析', '3级标题'],
        ['3.1.4 确定数据构成', '3级标题'],
        ['3.2 总体和抽样', '2级标题'],
        ['3.2.1 总体和个体', '3级标题'],
        ['3.2.2 样本', '3级标题'],
        ['第4章 Python基础', '1级标题'],
        ['4.1 Python的下载与安装', '2级标题']
        ['4.2 常用工具包的下载与安装, '2级标题']
        ['思考题', '2级标题']
    ]
    输入:"""
        prompt = prompt + '\n[\n'
        for line in directory_list:
            prompt += f" {line},\n"
        prompt += ']\n输出:\n'
        return prompt, 'You are a helpful assistant.'
