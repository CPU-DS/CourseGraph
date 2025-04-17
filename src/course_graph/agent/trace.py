# -*- coding: utf-8 -*-
# Create Date: 2025/03/31
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/agent/trace.py
# Description: 日志跟踪

from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, List
from datetime import datetime
from course_graph import set_logger, logger
from course_graph.agent import Agent

set_logger(console=True, file=False)


class TraceEventType(Enum):
    USER_MESSAGE = 'user_message'
    AGENT_MESSAGE = 'agent_message'
    AGENT_SWITCH = 'agent_switch'
    TOOL_CALL = 'tool_call'
    TOOL_RESULT = 'tool_result'
    CONTEXT_UPDATE = 'context_update'
    MCP_TOOL_CALL = 'mcp_tool_call'


@dataclass
class TraceEvent:
    timestamp: datetime
    agent: Agent
    event_type: TraceEventType
    data: dict


class Trace(TypedDict):
    trace_id: str
    events: List[TraceEvent]
    start_time: datetime
    end_time: datetime
