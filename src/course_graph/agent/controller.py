# -*- coding: utf-8 -*-
# Create Date: 2024/10/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/controller.py
# Description: 定义控制器

from .agent import Agent
import copy
import json
from .types import *
import inspect
from .exception import MaxTurnsException
import uuid
import time
from typing import Callable

class Controller:

    def __init__(self,
                 context_variables: ContextVariables | dict = None,
                 max_turns: int = 20,
                 trace_callback: Callable[[TraceEvent], None] = None) -> None:
        """ Agent 运行控制

        Args:
            context_variables (ContextVariables, optional): 上下文变量. Defaults to ContextVariables().
            max_turns (int, optional): 最大轮数. Defaults to 20.
            trace_callback (Callable[[TraceEvent], None], optional): 追踪回调, 传递 trace 事件. Defaults to None.
        """
        self.max_turns = max_turns
        match context_variables:
            case ContextVariables():
                self.context_variables = context_variables
            case _:  # dict or None
                self.context_variables = ContextVariables(context_variables)
        
        self.trace = Trace(
            trace_id=str(uuid.uuid4()), 
            events=[], start_time=time.time())
        self.trace_callback = trace_callback

    def set_agent_instruction(self, agent: Agent) -> None:
        """ 初始化 Agent instruction

        Args:
            agent (Agent): 新 Agent
        """
        match agent.instruction:
            case _ if callable(agent.instruction):
                parameters = inspect.signature(agent.instruction).parameters
                args = (self.context_variables,) if len(parameters) == 1 else ()
                agent.llm.instruction = agent.instruction(*args)
            case _:
                agent.llm.instruction = agent.instruction

    def __call__(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        return self.run(agent=agent, message=message)
    
    def _add_trace_event(self, event: TraceEvent) -> None:
        """ 添加trace事件 """
        self.trace['events'].append(event)
        if self.trace_callback:
            self.trace_callback(event)

    def run(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        """ 运行 Agent

        Args:
            agent (Agent): Agent对象
            message (str, optional): 用户输入. Defaults to None.

        Returns:
            (Agent, str): 最终激活的 Agent 和输出
        """
        turn = 1
        
        self.set_agent_instruction(agent)
        if not message:
            agent.add_assistant_message(agent.name)
            
        self._add_trace_event(TraceEventUserMessage(
            timestamp=time.time(),
            agent_name=agent.name,
            message=message
        ))
        
        assistant_output = agent.chat(message)
        
        self._add_trace_event(TraceEventAgentMessage(
            timestamp=time.time(),
            agent_name=agent.name,
            message=assistant_output.content
        ))
        
        while assistant_output.tool_calls:  # None 或者空数组
            functions = assistant_output.tool_calls
            for item in functions:
                function = item.function
                if (tool_function := agent.tool_functions.get(function.name)) is not None:
                    args = json.loads(function.arguments)

                    # 自动注入上下文变量
                    if (var_name := agent.use_context_variables.get(function.name)) is not None:
                        args[var_name] = self.context_variables

                    # 自动注入当前Agent
                    if (var_name := agent.use_agent_variables.get(function.name)) is not None:
                        args[var_name] = agent

                    self._add_trace_event(TraceEventToolCall(
                        timestamp=time.time(),
                        agent_name=agent.name,
                        function=function.name,
                        arguments=args
                    ))
                    
                    tool_content = tool_function(**args)
                    
                    match tool_content:
                        case Agent() as new_agent:
                            result = Result(agent=new_agent, content=json.dumps({'assistant': new_agent.name}, ensure_ascii=False))
                        case str() as content:
                            result = Result(content=content)
                        case ContextVariables() as new_variables:
                            result = Result(context_variables=new_variables)
                        case Result() as result:  # 上述三种返回值的组合类
                            pass
                        case _:
                            result = Result()
                    
                    self._add_trace_event(TraceEventToolResult(
                        timestamp=time.time(),
                        agent_name=agent.name,
                        function=function.name,
                        result=result
                    ))

                    agent.add_tool_call_message(result.content, item.id)
                    if result.agent is not None:  # 转移给其他Agent
                        if result.message:
                            result.agent.messages.extend(copy.deepcopy(agent.messages))
                        
                        self._add_trace_event(TraceEventAgentSwitch(
                            timestamp=time.time(),
                            agent_name=agent.name,
                            to_agent=result.agent.name
                        ))
                        agent = result.agent

                    self._add_trace_event(TraceEventContextUpdate(
                        timestamp=time.time(),
                        agent_name=agent.name,
                        old_context=self.context_variables,
                        new_context=result.context_variables
                    ))
                    self.context_variables.update(result.context_variables)
                    
                    self.set_agent_instruction(agent)

            assistant_output = agent.chat()
            turn += 1
            if turn > self.max_turns:
                raise MaxTurnsException
        
        message = assistant_output.content
        self._add_trace_event(TraceEventAgentMessage(
            timestamp=time.time(),
            agent_name=agent.name,
            message=message
        ))
        self.trace['end_time'] = time.time()

        return agent, assistant_output.content
