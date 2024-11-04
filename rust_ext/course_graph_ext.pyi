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


def structure_post_process(
        detections: list[tuple[str, tuple[float, ...]]],
        iou_threshold: float) -> list[tuple[str, tuple[float, ...]]]:
    """ 检测结果后处理

    Args:
        detections (list[tuple[str, tuple[float, ...]]]): 检测结果
        iou_threshold (float): iou 阈值

    Returns:
        list[tuple[str, tuple[float, ...]]]: 处理后检测结果
    """

def find_longest_consecutive_sequence(
        nums: list[int]) -> tuple[int, int]:
    """ 找到一个最长的连续序列的起点和终点

    Args:
        nums (list[int]): 序列

    Returns:
        tuple[int, int]: 起点数字和终点数字
    """