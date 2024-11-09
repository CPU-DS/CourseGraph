# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/llm/prompt.py
# Description: 定义提示词类

from abc import ABC, abstractmethod
import re
import json
from .prompt_strategy import ExamplePromptStrategy
from .ontology import ontology


class ExtractPrompt(ABC):

    def __init__(self) -> None:
        """ 信息抽取提示词类, 包含获取提示词和格式化模型返回两类方法
        """
        pass

    @abstractmethod
    def get_ner_prompt(self, content: str) -> tuple[str, str]:
        """ 获取实体抽取提示词

        Args:
            content (str): 待抽取的文本内容

        Returns:
            tuple[str, str]: 组合后的提示词, 指令
        """
        raise NotImplementedError

    @abstractmethod
    def get_re_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        """ 获取关系抽取的提示词

        Args:
            content (str): 待抽取的文本内容
            entities: (list[str]): 实体列表

        Returns:
            tuple[str, str]: 组合后的提示词, 指令
        """
        raise NotImplementedError

    @abstractmethod
    def get_ae_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        """ 获取属性抽取的提示词

        Args:
            content (str): 待抽取的文本内容
            entities: (list[str]): 实体列表

        Returns:
            tuple[str, str]: 组合后的提示词, 指令
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_attr_prompt(self, entity: str, attr: str,
                             values: list[str]) -> tuple[str, str]:
        """ 要求模型为实体的属性总结一个最佳的值

        Args:
            entity (str): 实体名称
            attr (str): 属性
            values (list[str]): 属性值列表

        Returns:
            tuple[str, str]: 组合后的提示词, 指令
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, response: str) -> list | dict:
        """ 将模型返回处理成列表或字典格式

        Args:
            response (str): 模型输出


        Returns:
            list | dict: 格式输出
        """
        raise NotImplementedError


class ExamplePrompt(ExtractPrompt):

    def __init__(self, strategy: ExamplePromptStrategy = None) -> None:
        """ 获取提取提示词, 使用多种提示词优化, 包括CoT、基于动态检索的ICL

        Args:
            strategy (ExamplePromptStrategy, optional): 动态检索提示词策略. Defaults to None.
        """
        super().__init__()
        self.strategy = strategy

    def get_ner_prompt(self, content: str) -> tuple[str, str]:
        if self.strategy is None:
            examples = [
                {
                    "input":
                        "如果有人问你现在有多幸福, 你会如何回答呢？一般的人可能会给出诸如“还可以吧”或者“不是那么幸福”等笼统的回答。如果有人回答“我现在的幸福指数是10.23”的话, 可能会把人吓一跳吧。因为他用一个数值指标来评判自己的幸福程度。这里的幸福指数只是打个比方, 实际上神经网络的学习也在做同样的事情。\n神经网络的学习通过某个指标表示现在的状态。然后, 以这个指标为基准, 寻找最优权重参数。和刚刚那位以幸福指数为指引寻找“最优人生”的人一样, 神经网络以某个指标为线索寻找最优权重参数。神经网络的学习中所用的指标称为损失函数（loss function）。这个损失函数可以使用任意函数, 但一般用均方误差和交叉熵误差等。\n损失函数是表示神经网络性能的“恶劣程度”的指标, 即当前的神经网络对监督数据在多大程度上不拟合, 在多大程度上不一致。以“性能的恶劣程度”为指标可能会使人感到不太自然, 但是如果给损失函数乘上一个负值, 就可以解释为“在多大程度上不坏”, 即“性能有多好”。并且, “使性能的恶劣程度达到最小”和“使性能的优良程度达到最大”是等价的, 不管是用“恶劣程度”还是“优良程度”, 做的事情本质上都是一样的。",
                    "output":
                        "这段文字介绍了我们一般通过损失函数来评价神经网络的性能, 可以了解到损失函数这一概念。抽取到的实体为```json\n{\"知识点\":[\"损失函数\"]}\n```"
                },
                {
                    "input":
                        "神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题, 解决这个问题的过程称为最优化（optimization）。遗憾的是, 神经网络的最优化问题非常难。这是因为参数空间非常复杂, 无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。\n而且, 在深度神经网络中, 参数的数量非常庞大, 导致最优化问题更加复杂。在前几章中, 为了找到最优参数, 我们将参数的梯度（导数）作为了线索。使用参数的梯度, 沿梯度方向更新参数, 并重复这个步骤多次, 从而逐渐靠近最优参数, 这个过程称为随机梯度下降法（stochastic gradient descent）, 简称SGD。\nSGD是一个简单的方法, 不过比起胡乱地搜索参数空间, 也算是“聪明”的方法。但是, 根据不同的问题, 也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点, 并介绍SGD以外的其他最优化方法。",
                    "output":
                        "这段文字介绍了神经网络的学习就是参数最优化的过程, 并且通常使用随机梯度下降法来寻找最优参数。抽取到的实体为```json\n{\"知识点\": [\"最优化\",\"随机梯度下降法\"]}\n```"
                },
                {
                    "input":
                        "顺便提一下, 在图3-2的网络中, 偏置b并没有被画出来。如果要明确地表示出b, 可以像图3-3那样做。图3-3中添加了权重为b的输入信号1。这个感知机将x1、x2、1三个信号作为神经元的输入, 将其和各自的权重相乘后, 传送至下一个神经元。在下一个神经元中, 计算这些加权信号的总和。\n如果这个总和超过0, 则输出1, 否则输出0。另外, 由于偏置的输入信号一直是1, 所以为了区别于其他神经元, 我们在图中把这个神经元整个涂成灰色。",
                    "output":
                        "这段文字可能在描述一张有关神经元的图像, 但是没有介绍一个新的概念或者引入新的名词, 所以没有能够抽取出来的知识点。所以返回为```json\n{}\n```"
                },
            ]
        else:
            examples = self.strategy.get_ner_example(content)
        prompt = {
            "instruction":
                "请对input的内容进行总结根据总结从中抽取出符合schema类型的实体。最后请给出你的总结和抽取到的类型以及对应的列表, 返回的格式为 ```json\n{\"entity_type1\": [\"entity1\", \"entity2\"]}\n```",
            "schema": ontology.entities,
            "examples": examples,
            "input": content
        }
        return json.dumps(prompt, indent=4, ensure_ascii=False), "你是专门进行实体抽取的专家"

    def get_re_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        if self.strategy is None:
            examples = [
                {
                    "input":
                        "实体列表为: [\"过拟合\", \"权值衰减\", \"权重均一化\"], 文本片段为: 后面我们会介绍抑制过拟合、提高泛化能力的技巧——权值衰减（weightdecay）。简单地说, 权值衰减就是一种以减小权重参数的值为目的进行学习的方法。通过减小权重参数的值来抑制过拟合的发生。\n如果想减小权重的值, 一开始就将初始值设为较小的值才是正途。实际上, 在这之前的权重初始值都是像0.01 * np.random.randn(10, 100)这样, 使用由高斯分布生成的值乘以0.01后得到的值（标准差为0.01的高斯分布）。\n如果我们把权重初始值全部设为0以减小权重的值, 会怎么样呢？从结论来说, 将权重初始值设为0不是一个好主意。事实上, 将权重初始值设为0的话, 将无法正确进行学习。\n为什么不能将权重初始值设为0呢？严格地说, 为什么不能将权重初始值设成一样的值呢？这是因为在误差反向传播法中, 所有的权重值都会进行相同的更新。比如, 在2层神经网络中, 假设第1层和第2层的权重为0。这样一来, 正向传播时, 因为输入层的权重为0, 所以第2层的神经元全部会被传递相同的值。第2层的神经元中全部输入相同的值, 这意味着反向传播时第2层的权重全部都会进行相同的更新（回忆一下“乘法节点的反向传播”\n的内容）。因此, 权重被更新为相同的值, 并拥有了对称的值（重复的值）这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化（严格地讲, 是为了瓦解权重的对称结构）, 必须随机生成初始值。",
                    "output":
                        "```这段文字介绍了抑制过拟合的方法之一权值衰减。抽取到的关系为```json\n[{\"head\": \"过拟合\", \"relation\": \"相关\", \"tail\": \"权值衰减\"}]\n```"
                },
                {
                    "input":
                        "实体列表为: [\"矩阵乘法\", \"np.dot()\"], 文本片段为: 下面, 我们来介绍矩阵（二维数组）的乘积。比如2 × 2的矩阵, 其乘积可以像图3-11这样进行计算（按图中顺序进行计算是规定好了的）。\n如本例所示, 矩阵的乘积是通过左边矩阵的行（横向）和右边矩阵的列（纵向）以对应元素的方式相乘后再求和而得到的。并且, 运算的结果保存为新的多维数组的元素。比如, A的第1行和B的第1列的乘积结果是新数组的第1行第1列的元素, A的第2行和B的第1列的结果是新数组的第2行第1列的元素。另外, 在本书的数学标记中, 矩阵将用黑斜体表示（比如, 矩阵A）, 以区别于单个元素的标量（比如, a或b）。这个运算在Python中可以用如下代码实现。\n>>> A = np.array([[1,2], [3,4]])\n>>> A.shape(2, 2)>>> B = np.array([[5,6], [7,8]])>>> B.shape(2, 2)>>> np.dot(A, B)array([[19, 22],       [43, 50]])这里, A 和B 都是2 × 2 的矩阵, 它们的乘积可以通过NumPy 的np.dot()函数计算（乘积也称为点积）。np.dot()接收两个NumPy数组作为参数, 并返回数组的乘积。这里要注意的是, np.dot(A, B)和np.dot(B, A)的值可能不一样。和一般的运算（+或*等）不同, 矩阵的乘积运算中, 操作数（A、B）的顺序不同, 结果也会不同。\n这里介绍的是计算2 × 2形状的矩阵的乘积的例子, 其他形状的矩阵的乘积也可以用相同的方法来计算。比如, 2 × 3的矩阵和3 × 2 的矩阵的乘积可按如下形式用Python来实现。\n>>> A = np.array([[1,2,3], [4,5,6]])>>> A.shape(2, 3)>>> B = np.array([[1,2], [3,4], [5,6]])>>> B.shape(3, 2)>>> np.dot(A, B)array([[22, 28],       [49, 64]])2 × 3的矩阵A和3 × 2的矩阵B的乘积可按以上方式实现。这里需要注意的是矩阵的形状（shape）。具体地讲, 矩阵A的第1维的元素个数（列数）必须和矩阵B的第0维的元素个数（行数）相等。在上面的例子中, 矩阵A的形状是2 × 3, 矩阵B的形状是3 × 2, 矩阵A的第1维的元素个数（3）和矩阵B的第0维的元素个数（3）相等。如果这两个值不相等, 则无法计算矩阵的乘积。比如, 如果用Python计算2 × 3 的矩阵A和2 × 2的矩阵C的乘积, 则会输出如下错误。\n>>> C = np.array([[1,2], [3,4]])>>> C.shape\n(2, 2)>>> A.shape(2, 3)>>> np.dot(A, C)Traceback (most recent call last):  File \"<stdin>\", line 1, in <module>ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)这个错误的意思是, 矩阵A的第1维和矩阵C的第0维的元素个数不一致（维度的索引从0开始）。也就是说, 在多维数组的乘积运算中, 必须使两个矩阵中的对应维度的元素个数一致, 这一点很重要。我们通过图3-12再来确认一下。\n这个错误的意思是, 矩阵A的第1维和矩阵C的第0维的元素个数不一致（维度的索引从0开始）。也就是说, 在多维数组的乘积运算中, 必须使两个矩阵中的对应维度的元素个数一致, 这一点很重要。我们通过图3-12再来确认一下。\n图3-12中, 3 × 2的矩阵A和2 × 4 的矩阵B的乘积运算生成了3 × 4的矩阵C。如图所示, 矩阵A和矩阵B的对应维度的元素个数必须保持一致。\n此外, 还有一点很重要, 就是运算结果的矩阵C的形状是由矩阵A的行数和矩阵B的列数构成的。\n另外, 当A是二维矩阵、B是一维数组时, 如图3-13所示, 对应维度的元素个数要保持一致的原则依然成立。\n可按如下方式用Python实现图3-13的例子。",
                    "output":
                        "这段文字介绍了Numpy的矩阵乘法, 以及使用 np.dot()实现矩阵乘法, 所以应该是先后学习的顺序关系。抽取到的关系为```json\n[{\"head\": \"矩阵乘法\", \"relation\": \"顺序\", \"tail\": \"np.dot()\"}]\n```"
                },
                {
                    "input":
                        "实体列表为: [\"阶跃函数\", \"sigmod函数\", \"线性函数\", \"非线性函数\"], 文本片段为: 阶跃函数和sigmoid函数还有其他共同点, 就是两者均为非线性函数。\nsigmoid函数是一条曲线, 阶跃函数是一条像阶梯一样的折线, 两者都属于非线性的函数。\n在介绍激活函数时, 经常会看到“非线性函数”和“线性函数”等术语。\n函数本来是输入某个值后会返回一个值的转换器。向这个转换器输入某个值后, 输出值是输入值的常数倍的函数称为线性函数（用数学式表示为h(x) = cx。c为常数）。因此, 线性函数是一条笔直的直线。\n而非线性函数, 顾名思义, 指的是不像线性函数那样呈现出一条直线的函数。\n神经网络的激活函数必须使用非线性函数。换句话说, 激活函数不能使用线性函数。为什么不能使用线性函数呢？因为使用线性函数的话, 加深神经网络的层数就没有意义了。\n线性函数的问题在于, 不管如何加深层数, 总是存在与之等效的“无隐藏层的神经网络”。为了具体地（稍微直观地）理解这一点, 我们来思考下面这个简单的例子。这里我们考虑把线性函数h(x) = cx 作为激活函数, 把y(x) = h(h(h(x))) 的运算对应3 层神经网络A。这个运算会进行y(x) = c × c × c × x的乘法运算, 但是同样的处理可以由y(x) = ax（注意, a = c 3）这一次乘法运算（即没有隐藏层的神经网络）来表示。如本例所示, 使用线性函数时, 无法发挥多层网络带来的优势。因此, 为了发挥叠加层所带来的优势, 激活函数必须使用非线性函数。",
                    "output": 
                        "```这段文字介绍了非线性函数以及线性函数, 二者是平等的相关关系, 以及非线性函数应该包含解决函数和sigmoid函数。抽取到的关系为```son\n[{\"head\": \"非线性函数\", \"relation\": \"相关\", \"tail\": \"线性函数\"}, {\"head\": \"非线性函数\", \"relation\": \"包含\", \"tail\": \"阶跃函数\"}, {\"head\": \"非线性函数\", \"relation\": \"包含\", \"tail\": \"sigmoid函数\"}]\n```"
                },
            ]
        else:
            examples = self.strategy.get_re_example(content)
        prompt = {
            "instruction":
                "请对输入的实体列表并根据已有文本片段, 一步步思考, 判断两两实体间的关系, 如果两两之间无关系或关系不在所指定的关系范围relations中, 则不返回。头尾实体不应该相同。返回为你的思考和关系三元组, 格式为 ```json\n[{\"head\": \"\", \"relation\": \"\", \"tail\": \"\"}]\n```",
            "relations": ontology.relations,
            "examples": examples,
            "input": f"实体列表为: {entities}, 文本片段为: '{content}'"
        }
        return json.dumps(prompt, indent=4, ensure_ascii=False), "你是专门进行关系判别的专家"

    def get_ae_prompt(self, content: str,
                      entities: list[str]) -> tuple[str, str]:
        if self.strategy is None:
            examples = [{
                "input":
                    """实体列表为: ['最优化', '随机梯度下降法'], 文本片段为: 神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题, 解决这个问题的过程称为最优化（optimization）。遗憾的是, 神经网络的最优化问题非常难。这是因为参数空间非常复杂, 无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。
                       而且, 在深度神经网络中, 参数的数量非常庞大, 导致最优化问题更加复杂。在前几章中, 为了找到最优参数, 我们将参数的梯度（导数）作为了线索。使用参数的梯度, 沿梯度方向更新参数, 并重复这个步骤多次, 从而逐渐靠近最优参数, 这个过程称为随机梯度下降法（stochastic gradient descent）, 
                       简称SGD。SGD是一个简单的方法, 不过比起胡乱地搜索参数空间, 也算是“聪明”的方法。但是, 根据不同的问题, 也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点, 并介绍SGD以外的其他最优化方法。""",
                "output":
                    "```json\n{\"最优化\": {\"定义\":\"寻找神经网络最优参数的过程\"}, \"随机梯度下降法\": {\"定义\":\"使用参数的梯度, 沿梯度方向更新参数, 并重复这个步骤多次, 从而逐渐靠近最优参数\"}}\n```"
            }]
        else:
            examples = self.strategy.get_ae_example(content)
        prompt = {
            "instruction":
                "请对输入的实体列表根据已有文本片段各自抽取他们的属性值。属性范围只能来源于提供的attributes, 属性值无需完全重复原文, 可以是你根据原文进行的总结, 如果实体没有能够总结的属性值则不返回。返回格式为 ```json\n{\"entity1\": {\"attribute1\":\"value\"}}\n```",
            "attributes": ontology.attributes,
            "examples": examples,
            "input": f"实体列表为: {entities}, 文本片段为: '{content}'"
        }
        return json.dumps(prompt, indent=4, ensure_ascii=False), "你是专门进行属性抽取的专家"

    def get_best_attr_prompt(self, entity: str, attr: str,
                             values: list[str]) -> tuple[str, str]:
        prompt = {
            "instruction":
                "请根据实体的属性对应的值列表, 总结出一个最佳的属性值。只需要返回总结的属性值即可。",
            "examples": [{
                "input": """实体为: 'Numpy', 属性为: '定义', 属性值列表为: [
                'NumPy提供了许多用于操作多维数组的便捷方法, 常与Python一起用于数据分析和科学计算。', 
                'NumPy是一个用于Python编程语言的科学计算库, 它提供了强大的N维数组对象, 以及大量的数学函数来操作这些数组。',
                '用于数值计算的库, 提供了很多高级的数学算法和便利的数组（矩阵）操作方法'
                ]""",
                "output": "NumPy是一个用于Python编程语言的科学计算库, 提供了许多用于操作多维数组的便捷方法, 常与Python一起用于数据分析和科学计算。"
            }],
            "input":
                f"实体为: '{entity}', 属性为: '{attr}', 属性值列表为: {values}"
        }
        return json.dumps(prompt, indent=4, ensure_ascii=False), "你是专门进行属性判别的专家"

    def post_process(self, response: str) -> list | dict | None:
        """ 将模型返回处理成列表或字典格式

        Args:
            response (str): 模型输出

        Returns:
            list | dict | None: 格式输出
        """
        replace_tuple = [('\\', ''), ('“', '"'), ('”', '"')]  # 替换掉可能出现的非法字符
        fragments = re.findall(r'```.*?\n([\s\S]*?)\n?```', response)
        if len(fragments) > 0:
            fragment: str = fragments[-1]  # 可能会返回多个结果从语义上只取最后一个结果
            for a, b in replace_tuple:
                fragment = fragment.replace(a, b)
            try:
                res = json.loads(fragment)
                return res
            except json.decoder.JSONDecodeError:
                return None
