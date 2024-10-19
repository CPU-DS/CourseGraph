# -*- coding: utf-8 -*-
# Create Date: 2024/10/17
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/agent_orchestration.py
# Description: 智能体编排：中断对话，用户介入

from course_graph.llm import Qwen
from course_graph.agent import Agent, Controller


def transfer_to_ceo():
    """ 将控制权转移到CEO。
    """
    return ceo


def transfer_to_product_manager():
    """ 将控制权转移到产品经理。
    """
    return product_manager


def transfer_to_coder():
    """ 将控制权转移到程序员。
    """
    return coder


def transfer_to_code_review():
    """ 将控制权转移到代码审查。
    """
    return code_review


model = Qwen()

ceo = Agent(
    name='CEO',
    llm=model,
    functions=[transfer_to_product_manager],
    instruction=
    '你是一家互联网公司的CEO，负责了解用户的意图并分析可行性。如果可行，请将控制权交给产品经理设计具体的需求项。在整个项目完成之后，接受整个团队的工作成功并返回给用户。'
)

product_manager = Agent(
    name='product_manager',
    llm=model,
    functions=[transfer_to_coder],
    instruction=
    '你是一家互联网公司的产品经理，负责将CEO的用户转换为具体需求，包括这个产品应该包含哪些功能，该使用什么技术实现等等。当你设计好了需求后请将控制权交给程序员开始编写代码。'
)

coder = Agent(
    name='coder',
    llm=model,
    functions=[transfer_to_code_review],
    instruction='你是一家互联网公司的程序员，负责将产品经理的需求用代码实现出来。当你完成后请将控制权代码审查验证你的代码。')

code_review = Agent(
    name='code_review',
    llm=model,
    functions=[transfer_to_ceo],
    instruction='你是一家互联网公司的代码审查员，负责审查程序员的代码是否能够完成产品经理的需求。如果能够通过审查，请将控制交给CEO。')


def pretty_print(message):
    print(
        f'Role: {message["role"]}\tName: {message.get("name", "")}\tContent: {message["content"]}'
    )


controller = Controller(agent=ceo, messages_observer=pretty_print)
controller.run(message='帮我写一个贪吃蛇游戏。')
while True:
    controller.run(message=input('You:'))
