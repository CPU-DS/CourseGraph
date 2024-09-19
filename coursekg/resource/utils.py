# -*- coding: utf-8 -*-
# Create Date: 2024/08/05
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/parser/utils.py
# Description: 工具函数

import os
from glob import glob


def check_os_windows() -> bool:
    """ 判断操作系统是否为windows

    Returns:
        bool: 操作系统是否为windows
    """
    return True if os.name == 'nt' else False


def pptx2imgs(path: str, cache_path: str) -> list[str]:
    """ 将pptx转换为图片

    Args:
        path (str): pptx文件路径
        cache_path (str): 图片缓存路径.

    Returns:
        list[str]: 排序后的图片列表
    """
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    if check_os_windows():
        from pptx_tools import utils
        utils.save_pptx_as_png(cache_path, path, overwrite_folder=True)
    else:
        import fitz
        os.system(
            f"libreoffice --headless --convert-to pdf --outdir {cache_path} {path} > libreoffice_convert.log 2>&1"
        )
        pdf_path = os.path.join(
            cache_path,
            os.path.basename(path).replace('.pptx', '.pdf'))
        idx = 0
        for page in fitz.open(pdf_path):
            page.get_pixmap().save(
                os.path.join(cache_path,
                             f"{os.path.basename(path)}_{idx}.png"))
            idx += 1
        os.remove(pdf_path)

    res = glob(f'{cache_path}/*')
    res.sort()

    return res
