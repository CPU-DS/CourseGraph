# -*- coding: utf-8 -*-
# Create Date: 2024/11/03
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/agent/workflow.py
# Description: 工作流编排

from course_graph.llm import Qwen
from course_graph.agent import Agent, ContextVariables, Controller
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint


model = Qwen()
news = """
Two people were taken to hospital after a "significant fire" broke out at the BAE Systems nuclear submarine shipyard in Barrow-in-Furness, police have said.
Emergency services were called to the site, where the UK's nuclear submarines are built, at about 00:44 GMT on Wednesday.
Those taken to hospital were said to be suffering from suspected smoke inhalation and have since been released, BAE confirmed, with everyone "accounted for".
Cumbria Police said there was "no nuclear risk". An investigation into the cause of the fire is under way.
Police said the Devonshire Dock Hall, the site's main building facility, was evacuated overnight.
The Ministry of Defence (MOD) said it was working closely with BAE and the emergency services following the fire."""


def get_translator_instruction(context_variables: ContextVariables):
    return f"You are a translator. Please translate the following text to Chinese: {context_variables['news']}"


def get_chinese_news_editor_instruction(context_variables: ContextVariables):
    return f"你是一个新闻编辑, 请根据提供的内容编写一份新闻稿。 新闻稿需要包含以下元素: 标题、正文。原始内容如下: {context_variables['trans']}"


def get_english_news_editor_instruction(context_variables: ContextVariables):
    return f"You are a news editor, please write a press release based on the content provided. The press release needs to contain the following elements: title, body. The original content is as follows: {context_variables['news']}"


translator = Agent(name='translator',
                   llm=model,
                   instruction=get_translator_instruction)
chinese_news_editor = Agent(name='chinese_news_editor',
                            llm=model,
                            instruction=get_chinese_news_editor_instruction)
english_news_editor = Agent(name='english_news_editor',
                            llm=model,
                            instruction=get_english_news_editor_instruction)

controller = Controller(context_variables={'news': news})


def chinese_news_write(controller: Controller) -> str:
    _, trans = controller.run_sync(agent=translator)
    controller.context_variables['trans'] = trans
    _, res = controller.run_sync(agent=chinese_news_editor)
    return res


def english_news_write(controller: Controller) -> str:
    _, res = controller.run_sync(agent=english_news_editor)
    return res


def workflow(controller):
    with ThreadPoolExecutor() as executor:
        work1 = executor.submit(chinese_news_write, controller)
        work2 = executor.submit(english_news_write, controller)

        ch_res: str = work1.result()
        en_res: str = work2.result()

    pprint(ch_res)
    pprint(en_res)


if __name__ == '__main__':
    workflow(controller)
