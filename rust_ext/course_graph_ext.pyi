def get_title_from_latex(latex: str) -> list[str]:
    """ 从 latex 源码中提取标题类型字符串

    Args:
        latex (str): latex 源码

    Returns:
        list[str]: 标题类型字符串
    """
    ...


def get_list_from_string(text: str) -> list:
    """ 括号匹配提取列表

    Args:
        text (str): 待提取字符串

    Returns:
        list: 列表
    """
    ...


def replace_linefeed(sentence: str, ignore_end: bool, replace: str) -> str:
    """ 移除句子的换行符

    Args:
        sentence (str): 句子
        ignore_end (bool): 忽略句末的换行符
        replace (str): 换行符替换对象

    Returns:
        str: 新句
    """
