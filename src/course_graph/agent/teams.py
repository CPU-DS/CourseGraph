# -*- coding: utf-8 -*-
# Create Date: 2024/10/14
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/teams.py
# Description: 定义团队

from .agent import Agent
from abc import ABC, abstractmethod
from .trace import TraceEvent
from typing import Callable, Union
from .types import MaxTurnsException, TimeOutException, MaxActiveException
from .controller import Controller
from .utils import async_to_sync
import copy
import time


class Terminator(ABC):
    def __init__(self):
        self._data = dict()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: dict):
        self._data = value
        if hasattr(self, 'terminators'):
            for terminator in self.terminators:
                terminator.data = value

    def update(self, data: dict):
        self._data.update(data)
        self.data = self._data  # 保证下游更新

    @abstractmethod
    def is_match(self) -> bool:
        raise NotImplementedError

    def __and__(self, other: 'Terminator') -> 'TerminatorGroup':
        return TerminatorGroup([self, other], operator='AND')

    def __or__(self, other: 'Terminator') -> 'TerminatorGroup':
        return TerminatorGroup([self, other], operator='OR')


class TerminatorGroup(Terminator):

    def __init__(self, terminators: list[Terminator] | list['TerminatorGroup'], operator: str = 'OR') -> None:
        super().__init__()
        self.terminators = terminators
        self.operator = operator

    def is_match(self) -> bool:
        if self.operator == 'AND':
            return all(terminator.is_match() for terminator in self.terminators)
        return any(terminator.is_match() for terminator in self.terminators)

    def __and__(self, other: Union[Terminator, 'TerminatorGroup']) -> 'TerminatorGroup':
        if isinstance(other, TerminatorGroup):
            return TerminatorGroup(self.terminators + other.terminators, operator='AND')
        return TerminatorGroup(self.terminators + [other], operator='AND')

    def __or__(self, other: Union[Terminator, 'TerminatorGroup']) -> 'TerminatorGroup':
        if isinstance(other, TerminatorGroup):
            return TerminatorGroup(self.terminators + other.terminators, operator='OR')
        return TerminatorGroup(self.terminators + [other], operator='OR')


class TextTerminator(Terminator):
    def __init__(self, text: str) -> None:
        """
        当团队中任意一个 `Agent` 的响应中包含指定的文本时，团队将停止运行。

        Args:
            text (str): 指定的文本
        """
        super().__init__()
        self.text = text

    def is_match(self) -> bool:
        return self.text in self.data.get('message', '')


class MaxTurnsTerminator(Terminator):
    def __init__(self, max_turns: int) -> None:
        """
        最大轮数终止, 包含每个 `Agent` 调用工具的次数。

        Args:
            max_turns (int): 最大轮数
        """
        super().__init__()
        self.max_turns = max_turns

    def is_match(self) -> bool:
        if r := self.data.get('turns', 0) >= self.max_turns:
            raise MaxTurnsException
        return r


class MaxActiveTerminator(Terminator):
    def __init__(self, max_turns: int) -> None:
        """
        最大激活次数终止, 表示团队中能够激活 `Agent` 的最大次数。

        Args:
            max_turns (int): 最大激活次数
        """
        super().__init__()
        self.max_turns = max_turns

    def is_match(self) -> bool:
        if r := self.data.get('active_turns', 0) >= self.max_turns:
            raise MaxActiveException
        return r


class TimeOutTerminator(Terminator):
    def __init__(self, timeout: int) -> None:
        """
        超时终止，表示整个团队运行的最长时长。

        Args:
            timeout (int): 超时时间
        """
        super().__init__()
        self.start_time = time.time()
        self.timeout = timeout

    def is_match(self) -> bool:
        if r := time.time() - self.start_time > self.timeout:
            raise TimeOutException
        return r


class Team(ABC):

    def __init__(self, agents: list[Agent]) -> None:
        self.agents = agents
        self.controller = Controller()
        self.task = None
        self.terminator: Terminator | None = None
        self.global_messages = []

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError

    def run_sync(self) -> None:
        async_to_sync(self.run)()

    def reset(self) -> None:
        """
        重置团队
        """
        self.agents = []
        self.task = None
        self.terminator = None
        self.global_messages = []

    def add_task(self, task: str) -> None:
        """
        添加任务
        """
        self.task = task

    def add_trace_callback(self, callback: Callable[[TraceEvent], None]) -> None:
        """
        添加 Controller 的 Trace 回调
        
        Args:
            callback (Callable[[TraceEvent], None]): 回调函数
        """
        self.controller.trace_callback = callback

    def _check_terminator(self, data: dict) -> None:
        if self.terminator:
            self.terminator.update(data)
            if self.terminator.is_match():
                return


class RoundTeam(Team):
    def __init__(self, agents: list[Agent]) -> None:
        """
        初始化一个轮询团队
        
        Args:
            agents (list[Agent]): 团队中的 Agent 列表
        """
        super().__init__(agents)

    def __or__(self, other: Agent) -> 'RoundTeam':
        self.agents.append(other)
        return self

    async def run(self) -> None:
        """
        运行团队
        """
        if self.task:
            self.global_messages.append({
                'role': 'user',
                'content': self.task
            })
        while True:
            for agent in self.agents:
                agent.messages = copy.deepcopy(self.global_messages)
                response = await self.controller.run(agent)

                self._check_terminator({
                    'message': response.message,
                    'turns': self.terminator.data.get('turns', 0) + response.turns,
                    'active_turns': self.terminator.data.get('active_turns', 0) + 1
                })

                self.global_messages.append({
                    'role': 'assistant',
                    'content': response.message,
                    'name': agent.name
                })


def __or__(self, other: Agent) -> 'RoundTeam':
    return RoundTeam([self, other])


Agent.__or__ = __or__
