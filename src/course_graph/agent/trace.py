# -*- coding: utf-8 -*-
# Create Date: 2025/03/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/trace.py
# Description: 日志跟踪

from enum import Enum
from dataclasses import dataclass, field
from typing import TypedDict, List
from datetime import datetime
from course_graph import set_logger, logger
from course_graph.agent import Agent

set_logger(console=True, file=False)


class TraceEventType(Enum):
    USER_MESSAGE = 'user_message'
    AGENT_THINK = 'agent_think'
    AGENT_MESSAGE = 'agent_message'
    AGENT_SWITCH = 'agent_switch'
    TOOL_CALL = 'tool_call'
    TOOL_RESULT = 'tool_result'
    CONTEXT_UPDATE = 'context_update'
    MCP_TOOL_CALL = 'mcp_tool_call'


@dataclass
class TraceEvent:
    agent: Agent
    event_type: TraceEventType
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)


class Trace(TypedDict):
    trace_id: str
    events: List[TraceEvent]
    start_time: datetime
    end_time: datetime


def trace_callback(trace: TraceEvent) -> None:
    """ 默认 Trace 回调
    """
    data = ""
    match trace.event_type:
        case TraceEventType.USER_MESSAGE | TraceEventType.AGENT_MESSAGE | TraceEventType.AGENT_THINK:
            data = trace.data['message']
        case TraceEventType.TOOL_CALL | TraceEventType.MCP_TOOL_CALL:
            args = ", ".join([f"{k}='{v}'" for k,v in trace.data['arguments'].items()])
            data = f"{trace.data['function']}({args})"
        case TraceEventType.TOOL_RESULT:
            result = trace.data['result']
            data = result['content']
            if result.get('context_variables'):
                data += f" \t| {result['context_variables']}"
            if result.get('message'):
                data += f" \t| message={result['message']}"
        case TraceEventType.CONTEXT_UPDATE:
            data = trace.data['context']
        case TraceEventType.AGENT_SWITCH:
            data = f"{trace.agent.name}({trace.agent.id}) -> {trace.data['to_agent'].name}({trace.data['to_agent'].id})"
        case _:
            pass
    logger.info(f"{trace.agent.name}({trace.agent.id})\t| {trace.event_type.name}\t| {data}")
