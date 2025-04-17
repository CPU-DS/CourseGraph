# -*- coding: utf-8 -*-
# Create Date: 2024/10/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/controller.py
# Description: 定义控制器

from .agent import Agent
import copy
import json
from .types import Result, ContextVariables
from .trace import Trace, TraceEvent, TraceEventType
import inspect
from .exception import MaxTurnsException
from datetime import datetime
import uuid
from .utils import async_to_sync
from mcp.types import TextContent, ImageContent, EmbeddedResource, BlobResourceContents
from typing import Callable, Any


class Controller:

    def __init__(self,
                 context_variables: ContextVariables | dict = None,
                 max_turns: int = 20,
                 trace_callback: Callable[[TraceEvent], Any] = None) -> None:
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
            events=[],
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        self.trace_callback = trace_callback

    def set_agent_instruction(self, agent: Agent) -> None:
        """ 初始化 Agent instruction

        Args:
            agent (Agent): 新 Agent
        """
        match agent.instruction:
            case _ if callable(agent.instruction):
                parameters = inspect.signature(agent.instruction).parameters
                args = {}
                for arg_name, p in parameters.items():
                    if p.annotation == ContextVariables:
                        args[arg_name] = self.context_variables
                    else:
                        args[arg_name] = agent.instruction_args.get(arg_name, p.default)
                agent.llm.instruction = agent.instruction(**args)
            case _:
                agent.llm.instruction = agent.instruction

    def __call__(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        return self.run_sync(agent=agent, message=message)

    def _add_trace_event(self, event: TraceEvent) -> None:
        """ 添加trace事件 """
        self.trace['events'].append(event)
        if self.trace_callback:
            self.trace_callback(event)

    async def run(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
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

        self._add_trace_event(TraceEvent(
            timestamp=datetime.now(),
            event_type=TraceEventType.USER_MESSAGE,
            agent=agent,
            data={'message': message}
        ))

        assistant_output = agent.chat_completion(message)    

        while assistant_output.tool_calls:  # None 或者空数组
            functions = assistant_output.tool_calls
            for item in functions:
                function = item.function
                args = json.loads(function.arguments)

                if (tool_function := agent.tool_functions.get(function.name)) is not None:
                    
                    self._add_trace_event(TraceEvent(
                        timestamp=datetime.now(),
                        event_type=TraceEventType.TOOL_CALL,
                        agent=agent,
                        data={'function': function.name, 'arguments': args}
                    ))

                    # 自动注入上下文变量
                    if (var_name := agent.use_context_variables.get(function.name)) is not None:
                        args[var_name] = self.context_variables

                    # 自动注入当前Agent
                    if (var_name := agent.use_agent_variables.get(function.name)) is not None:
                        args[var_name] = agent

                    tool_content = tool_function(**args)
                    if inspect.iscoroutine(tool_content):  # 处理异步函数
                        tool_content = await tool_content

                    match tool_content:
                        case Agent() as new_agent:
                            result = Result(agent=new_agent,
                                            content=json.dumps({'assistant': new_agent.name}, ensure_ascii=False))
                        case str() as content:
                            result = Result(content=content)
                        case ContextVariables() as new_variables:
                            result = Result(context_variables=new_variables)
                        case Result() as result:  # 上述三种返回值的组合类
                            pass
                        case _:
                            result = Result()

                elif (mcp_sever := agent.mcp_functions.get(function.name)) is not None:
                    self._add_trace_event(TraceEvent(
                        timestamp=datetime.now(),
                        agent=agent,
                        event_type=TraceEventType.MCP_TOOL_CALL,
                        data={'function': function.name, 'arguments': args}
                    ))

                    resp = (await mcp_sever.session.call_tool(function.name, args)).content
                    text_contents = []
                    for content in resp:
                        match content:
                            case TextContent():
                                text_contents.append(content.text)
                            case ImageContent():
                                text_contents.append(content.data)
                            case EmbeddedResource():
                                text_contents.append(content.resource.text)
                            case BlobResourceContents():
                                text_contents.append(content.blob)
                    text = '\n'.join(text_contents)
                    result = Result(content=text)
                else:
                    result = Result(content=f'Failed to call tool: {function.name}')

                trace_result = {'content': result.content}
                if result.context_variables._vars:
                    trace_result['context_variables'] = result.context_variables._vars
                if not result.message:
                    trace_result['message'] = False
                    
                self._add_trace_event(TraceEvent(
                    timestamp=datetime.now(),
                    agent=agent,
                    event_type=TraceEventType.TOOL_RESULT,
                    data={'function': function.name, 'result': trace_result}
                ))

                agent.add_tool_call_message(result.content, item.id)
                if result.agent is not None:  # 转移给其他Agent
                    if result.message:
                        result.agent.messages.extend(copy.deepcopy(agent.messages))

                    self._add_trace_event(TraceEvent(
                        timestamp=datetime.now(),
                        agent=agent,
                        event_type=TraceEventType.AGENT_SWITCH,
                        data={'to_agent': result.agent.name}
                    ))
                    agent = result.agent
                if result.context_variables._vars:
                    self._add_trace_event(TraceEvent(
                        timestamp=datetime.now(),
                        agent=agent,
                        event_type=TraceEventType.CONTEXT_UPDATE,
                        data={'old_context': self.context_variables, 'new_context': result.context_variables}
                    ))
                    self.context_variables.update(result.context_variables)

                self.set_agent_instruction(agent)

            assistant_output = agent.chat_completion()
            turn += 1
            if turn > self.max_turns:
                raise MaxTurnsException

        self._add_trace_event(TraceEvent(
            timestamp=datetime.now(),
            event_type=TraceEventType.AGENT_MESSAGE,
            agent=agent,
            data={'message': assistant_output.content}
        ))

        self.trace['end_time'] = datetime.now()

        return agent, assistant_output.content

    def run_sync(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        """ 运行 Agent (同步版本)

        Args:
            agent (Agent): Agent对象
            message (str, optional): 用户输入. Defaults to None.

        Returns:
            (Agent, str): 最终激活的 Agent 和输出
        """
        return async_to_sync(self.run)(agent, message)
    