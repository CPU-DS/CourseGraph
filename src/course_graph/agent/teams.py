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
from dataclasses import dataclass
from .types import Tool, Result


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


@dataclass
class TeamResponse:
    ...


class Team(ABC):

    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.controller = Controller()
        self.termination: Termination | None = None
        
    @property
    def trace(self):
        return self.controller.trace

    @abstractmethod
    async def run(self, task: str) -> TeamResponse:
        raise NotImplementedError

    def run_sync(self, task: str) -> TeamResponse:
        return async_to_sync(self.run)(task)

    def reset(self) -> None:
        """
        重置团队
        """
        for agent in self.agents:
            agent.messages = []

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
        global_messages = [{
            'role': 'user',
            'content': task
        }]
        while True:
            for agent in self.agents:
                agent.messages = global_messages
                response = await self.controller.run(agent)
                if self._check_termination({
                    'message': response.message
                }):
                    return TeamResponse()
                global_messages.extend(copy.deepcopy(agent.messages))
                

class LinearTeam(Team):
    def __init__(self, agents: list[Agent]):
        """
        初始化一个线性团队
        
        Args:
            agents (list[Agent]): 团队中的 Agent 列表
        """
        super().__init__(agents)

    def __ge__(self, other: Agent) -> 'LinearTeam':
        self.agents.append(other)
        return self
    
    async def run(self, task: str) -> TeamResponse:
        """
        运行团队
        
        Args:
            task (str): 任务
        """
        for agent in self.agents:
            agent.add_user_message(task)
        response = None
        for agent in self.agents:
            if response:
                agent.add_assistant_message(response.message, response.agent.name)
            response = await self.controller.run(agent)
            if self._check_termination({
                    'message': response.message
                }):
                return TeamResponse()


class LeaderTeam(Team):
    def __init__(self, leader: Agent, subordinates: list[Agent]):
        """
        初始化一个领导团队
        
        Args:
            leader (Agent): 领导
            subordinates (list[Agent]): 跟随者
        """
        super().__init__([leader, *subordinates])
        self.leader = leader
        self.subordinates = subordinates
        self._descs: list[str] | tuple[str, ...] = []
        
        for subordinate in self.subordinates:
            
            def transfer(subordinate: Agent, instruction: str):
                subordinate.add_user_message(instruction)
                return Result(
                    agent=subordinate,
                    message=False
                )
            
            transfer_to_subordinate: Tool = {
                'function': lambda instruction, subordinate=subordinate: transfer(subordinate, instruction),  # avoid 延迟绑定
                'tool': {
                    'type': 'function',
                    'function': {
                        'name': f'transfer_to_{subordinate.name.replace(" ", "_")}',
                        'description': f"Transfer task to {subordinate.name}, who is instructed to '{subordinate.instruction}' and will continue the task.",
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'instruction': {
                                    'type': 'string',
                                    'description': f"The instruction for {subordinate.name} to continue the task."
                                }
                            },
                            'required': ['instruction']
                        }
                    }
                }
            }
            self.leader.add_tools(transfer_to_subordinate)
    
    def __le__(self, descs: list[str] | tuple[str, ...]) -> 'LeaderTeam':
        self.descs = descs
        return self
    
    @property
    def descs(self):
        return self._descs
    
    @descs.setter
    def descs(self, descs: list[str] | tuple[str, ...]):
        if len(descs) != len(self.subordinates):
            raise ValueError
        for idx, desc in enumerate(descs):
            self.leader.tools[idx]['function']['description'] = desc
        self._descs = descs
        
    async def run(self, task: str) -> TeamResponse:
        """
        运行团队
        
        Args:
            task (str): 任务
        """
        self.leader.add_user_message(task)
            
        while True:
            response = await self.controller.run(self.leader)
            if self._check_termination({
                    'message': response.message
                }):
                return TeamResponse()
            for subordinate in self.subordinates:
                if subordinate.messages:
                    self.leader.add_assistant_message(subordinate.messages[-1].content, subordinate.name)
                    subordinate.messages = []


def __or__(self, other: Agent) -> 'RoundTeam':
    return RoundTeam([self, other])

def __ge__(self, other: Agent) -> 'LinearTeam':
    return LinearTeam([self, other])

def __le__(self, other: list[Agent]) -> 'LeaderTeam':
    return LeaderTeam(leader=self, subordinates=other)

Agent.__or__ = __or__
Agent.__ge__ = __ge__
Agent.__le__ = __le__
