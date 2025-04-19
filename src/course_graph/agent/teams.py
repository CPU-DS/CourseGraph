# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/teams.py
# Description: 定义团队

from .agent import Agent
from abc import ABC, abstractmethod
from .trace import TraceEvent
from typing import Callable, Union
from .controller import Controller
from .utils import async_to_sync
import copy
import time
from dataclasses import dataclass


class Termination(ABC):
    def __init__(self):
        self._data = dict()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: dict):
        self._data = value
        if hasattr(self, 'terminations'):
            for termination in self.terminations:
                termination.data = value

    def update(self, data: dict):
        self._data.update(data)
        self.data = self._data  # 保证下游更新

    @abstractmethod
    def is_match(self) -> bool:
        raise NotImplementedError

    def __and__(self, other: 'Termination') -> 'TerminationGroup':
        return TerminationGroup([self, other], operator='AND')

    def __or__(self, other: 'Termination') -> 'TerminationGroup':
        return TerminationGroup([self, other], operator='OR')


class TerminationGroup(Termination):

    def __init__(self, terminations: list[Termination] | list['TerminationGroup'], operator: str = 'OR') -> None:
        super().__init__()
        self.terminations = terminations
        self.operator = operator

    def is_match(self) -> bool:
        if self.operator == 'AND':
            return all(termination.is_match() for termination in self.terminations)
        return any(termination.is_match() for termination in self.terminations)

    def __and__(self, other: Union[Termination, 'TerminationGroup']) -> 'TerminationGroup':
        if isinstance(other, TerminationGroup):
            return TerminationGroup(self.terminations + other.terminations, operator='AND')
        return TerminationGroup(self.terminations + [other], operator='AND')

    def __or__(self, other: Union[Termination, 'TerminationGroup']) -> 'TerminationGroup':
        if isinstance(other, TerminationGroup):
            return TerminationGroup(self.terminations + other.terminations, operator='OR')
        return TerminationGroup(self.terminations + [other], operator='OR')


class TextMentionTermination(Termination):
    def __init__(self, text: str) -> None:
        """
        当团队中任意一个 `Agent` 的响应中包含指定的文本时，团队将停止运行

        Args:
            text (str): 指定的文本
        """
        super().__init__()
        self.text = text

    def is_match(self):
        return self.text in self.data.get('message', '')


class MaxActiveTermination(Termination):
    def __init__(self, count: int) -> None:
        """
        最大激活次数终止, 表示团队中能够激活 `Agent` 的最大次数

        Args:
            count (int): 最大激活次数
        """
        super().__init__()
        self.count = count

    def is_match(self) -> bool:
        return self.data.get('active_count', 0) >= self.count


class TimeOutTermination(Termination):
    def __init__(self, timeout: int) -> None:
        """
        超时终止，表示整个团队运行的最长时长

        Args:
            timeout (int): 超时时间
        """
        super().__init__()
        self.start_time = time.time()
        self.timeout = timeout

    def is_match(self) -> bool:
        return time.time() - self.start_time > self.timeout


@dataclass
class TeamResponse:
    ...


class Team(ABC):

    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.controller = Controller()
        self.termination: Termination | None = None
        self.global_messages = []

    @abstractmethod
    async def run(self, task: str) -> TeamResponse:
        raise NotImplementedError

    def run_sync(self, task: str) -> TeamResponse:
        return async_to_sync(self.run)(task)

    def reset(self) -> None:
        """
        重置团队
        """
        self.global_messages = []

    def set_trace_callback(self, callback: Callable[[TraceEvent], None]) -> None:
        """
        设置 Controller 的 Trace 回调
        
        Args:
            callback (Callable[[TraceEvent], None]): 回调函数
        """
        self.controller.trace_callback = callback

    def _check_termination(self, data: dict) -> bool:
        if self.termination:
            self.termination.update(data)
            return self.termination.is_match()


class RoundTeam(Team):
    def __init__(self, agents: list[Agent]):
        """
        初始化一个轮询团队
        
        Args:
            agents (list[Agent]): 团队中的 Agent 列表
        """
        super().__init__(agents)

    def __or__(self, other: Agent) -> 'RoundTeam':
        self.agents.append(other)
        return self

    async def run(self, task: str) -> TeamResponse:
        """
        运行团队
        
        Args:
            task (str): 任务
        """
        self.global_messages.append({
            'role': 'user',
            'content': task
        })
        while True:
            for agent in self.agents:
                agent.messages = copy.deepcopy(self.global_messages)
                response = await self.controller.run(agent)
                if self._check_termination({
                    'message': response.message,
                    'active_count': self.termination.data.get('active_count', 0) + 1
                }):
                    return TeamResponse()
                self.global_messages.extend(agent.messages)
                

class LinearTeam(Team):
    def __init__(self, agents: list[Agent]):
        """
        初始化一个线性团队
        
        Args:
            agents (list[Agent]): 团队中的 Agent 列表
        """
        super().__init__(agents)

    def __gt__(self, other: Agent) -> 'LinearTeam':
        self.agents.append(other)
        return self
    
    async def run(self, task: str) -> TeamResponse:
        """
        运行团队
        
        Args:
            task (str): 任务
        """
        self.global_messages.append({
            'role': 'user',
            'content': task
        })
        for agent in self.agents:
            agent.add_user_message(task)
        
        response = None
        for agent in self.agents:
            if response:
                agent.add_assistant_message(response.message, response.agent.name)
            response = await self.controller.run(agent)
            self.global_messages.extend(agent.messages)
            
            

def __or__(self, other: Agent) -> 'RoundTeam':
    return RoundTeam([self, other])

def __gt__(self, other: Agent) -> 'LinearTeam':
    return LinearTeam([self, other])

Agent.__or__ = __or__
Agent.__gt__ = __gt__
