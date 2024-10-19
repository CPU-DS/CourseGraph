# -*- coding: utf-8 -*-
# Create Date: 2024/10/16
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/agent_orchestration.py
# Description: 智能体编排

from course_graph.llm import Qwen
from course_graph.agent import Agent, Controller, Result
from pprint import pprint


def weather(locations: str):
    """ 根据用户提供的地点查询当前天气。
    Args:
        locations (str): 地点
    """
    return Result(agent=core_agent,
                  content=f" {locations} 当前的天气是多云, 温度22-27摄氏度。")


def location():
    """ 查询用户当前所在的位置。
    """
    return "您当前所处的位置是北京市。"


def schedule_details():
    """ 查询用户当前的日程信息。
    """
    return Result(agent=core_agent, content=f"今天有一场考试, 时间是晚上9-10点。")


def add_alarm_clock(time: str):
    """ 帮用户定一个闹钟。

    Args:
        time (str): 闹钟时间。
    """
    return Result(agent=core_agent, content=f" {time} 的闹钟已经添加成功。")


def transfer_to_weather_agent():
    """ 将身份切换到处理天气查询的智能体上。
    """
    return weather_agent


def transfer_to_schedule_agent():
    """ 将身份切换到处理用户日程信息查询的智能体上。
    """
    return schedule_agent


def transfer_to_alarm_clock_agent():
    """ 将身份切换到帮用户定闹钟的智能体上。
    """
    return alarm_clock_agent


model = Qwen()
core_agent = Agent(
    name='core agent',
    llm=model,
    functions=[
        transfer_to_weather_agent, transfer_to_schedule_agent,
        transfer_to_alarm_clock_agent
    ],
    instruction=
    '你是一个核心智能体, 负责判断用户意图并切换到相应的智能体上执行任务。 注意, 一次只允许切换一种身份。 最后总结你所获取到的所有信息,并回答我的问题。'
)
weather_agent = Agent(name='weather agent',
                      llm=model,
                      functions=[weather, location],
                      instruction='你是一个负责天气查询的智能体。')
schedule_agent = Agent(name='schedule agent',
                       llm=model,
                       functions=[schedule_details],
                       instruction='你是一个负责用户日程信息查询的智能体。')
alarm_clock_agent = Agent(name='alarm clock agent',
                          llm=model,
                          functions=[add_alarm_clock],
                          instruction='你是一个负责帮用户定闹钟的智能体。')

controller = Controller(agent=core_agent)
resp = controller.run(
    message='帮我查询一下我这里的天气, 并查询一下我的日程信息。如果日程中有考试的话, 请帮我定一个闹钟, 时间是考试开始前的一个小时。')
pprint(resp)
pprint(controller.messages)
