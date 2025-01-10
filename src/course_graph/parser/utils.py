# -*- coding: utf-8 -*-
# Create Date: 2024/11/04
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/utils.py
# Description: 工具

import copy
from functools import wraps

def instance_method_transactional(*instance_variables):
    """ 装饰实例方法, 指定实例属性名称, 在方法抛出异常的时候回滚对这些属性的更改, 然后继续抛出异常。

    Args:
        *variables (str): 需要回滚的变量名称
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            original_state = {var: copy.deepcopy(getattr(self, var)) for var in instance_variables}
            
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                for var, value in original_state.items():
                    setattr(self, var, value)
                raise e
        return wrapper
    return decorator