# -*- coding: utf-8 -*-
# Create Date: 2024/10/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/controller.py
# Description: 定义控制器

from .agent import Agent
import copy
import json
from .types import Result, ContextVariables
from typing import Callable


class Controller:

    def __init__(self,
                 context_variables: ContextVariables | dict = None) -> None:
        """ Agent 运行控制

        Args:
            context_variables (ContextVariables, optional): 上下文变量. Defaults to ContextVariables().
        """
        match context_variables:
            case ContextVariables():
                self.context_variables = context_variables
            case _:  # dict or None
                self.context_variables = ContextVariables(context_variables)

    def set_agent_instruction(self, agent: Agent) -> None:
        """ 初始化 Agent instruction

        Args:
            agent (Agent): 新 Agent
        """
        match agent.instruction:
            case str() as instruction:
                pass
            case _:
                instruction = agent.instruction(self.context_variables)
        agent.llm.instruction = instruction

    def __call__(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        return self.run(agent=agent, message=message)

    def run(self, agent: Agent, message: str = None) -> tuple[Agent, str]:
        """ 运行 Agent

        Args:
            agent (Agent): Agent对象
            message (str, optional): 用户输入. Defaults to None.

        Returns:
            (Agent, str): Agent 和他最终的输出
        """
        self.set_agent_instruction(agent)
        if message is None:
            agent.add_assistant_message(agent.name)
        assistant_output = agent.chat(message)
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

                    tool_content = tool_function(**args)
                    match tool_content:
                        case Agent() as new_agent:
                            result = Result(agent=new_agent, content=json.dumps({'assistant': new_agent.name}))
                        case str() as content:
                            result = Result(content=content)
                        case ContextVariables() as new_variables:
                            result = Result(context_variables=new_variables)
                        case Result() as result:  # 上述三种返回值的组合类
                            pass
                        case _:
                            result = Result()

                    agent.add_tool_call_message(result.content, item.id)
                    if result.agent is not None:
                        if result.messages:
                            result.agent.messages.extend(copy.deepcopy(agent.messages))
                        agent = result.agent
                    # 更新上下文变量
                    self.context_variables.update(result.context_variables)

            assistant_output = agent.chat()

        return agent, assistant_output.content
