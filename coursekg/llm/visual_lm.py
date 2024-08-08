from .config import VisualConfig
import torch
from modelscope import AutoModel, AutoTokenizer


class MiniCPM:

    def __init__(self, path: str,
                 config: VisualConfig = VisualConfig()) -> None:
        """ MiniCPM系列模型, 执行图片问答任务

        Args:
            path (str): 模型名称或路径
            config (VisualConfig, optional): 配置. Defaults to VisualConfig().
        """
        self.model = AutoModel.from_pretrained(path,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16)
        self.model = self.model.to(device='cuda')

        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.model.eval()
        self.config = config

    def chat(self, msgs: list, sys_prompt: str = None) -> str:
        """ 图片问答

        Args:
            msgs (list): 输入内容.
            sys_prompt (str, optional): 系统提示词. Defaults to None.

        Returns:
            str: 模型输出
        """

        return self.model.chat(image=None,
                               msgs=msgs,
                               tokenizer=self.tokenizer,
                               sampling=True,
                               temperature=self.config.temperature,
                               sys_prompt=sys_prompt)
