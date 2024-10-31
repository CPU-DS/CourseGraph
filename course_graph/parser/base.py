# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/base.py
# Description: 定义文档、书签以及抽取知识图谱相关类和方法

from dataclasses import dataclass, field
from ..llm import LLM, ExtractPrompt, ExamplePrompt
import uuid
from loguru import logger
from .config import Config
from collections import Counter
from typing import TYPE_CHECKING
import random
import pickle
import os

if TYPE_CHECKING:
    from .parser import Parser
    from ..resource import ResourceMap, Resource, Slice


@dataclass
class KPEntity:
    id: str
    name: str
    relations: list['KPRelation'] = field(default_factory=list)
    attributes: dict[str, list] | dict[str, str] = field(
        default_factory=dict)  # 同一个属性可能会存在多个属性值, 后续选择一个最好的值
    resourceSlices: list['Slice'] = field(default_factory=list)


@dataclass
class KPRelation:
    id: str
    type: str
    tail: KPEntity
    attributes: dict[str, list] | dict[str, str] = field(default_factory=dict)


@dataclass
class PageIndex:
    index: int
    anchor: tuple[float, float]  # 等价于 bbox 中的 x1,y1


@dataclass
class BookMark:
    """ 书签
    """
    id: str
    title: str
    page_start: PageIndex
    page_end: PageIndex
    level: int
    subs: list['BookMark'] | list[KPEntity]
    resource: list['Resource']

    def set_page_end(self, page_end: PageIndex) -> None:
        """ 设置书签的结束页, 和直接修改 BookMark 对象的 page_end 属性不同, 该方法会考虑到书签嵌套的情况

        Args:
            page_end (int): 结束页
        """
        self.page_end = page_end
        if self.subs and isinstance(self.subs[-1], BookMark):
            self.subs[-1].set_page_end(page_end)

    def get_kps(self) -> list[KPEntity]:
        """ 获取当前书签下的所有 知识点实体

        Returns:
            list[KPEntity]: 实体列表
        """
        kps: list[KPEntity] = []

        def get_kp(bookmarks: list[BookMark]):
            for bk in bookmarks:
                if bk.subs and isinstance(bk.subs[-1], KPEntity):
                    kps.extend(bk.subs)
                else:
                    get_kp(bk.subs)

        get_kp([self])
        return kps


@dataclass
class Document:
    """文档
    """
    id: str
    name: str
    file_path: str
    bookmarks: list[BookMark]
    parser: 'Parser'
    knowledgepoints: list[KPEntity] = field(default_factory=list)  # 全局共享状态

    @classmethod
    def from_parser(cls, parser: 'Parser') -> 'Document':
        """ 使用 Parser 对象生成 Document 对象

        Args:
            parser (Parser): 解析器

        Returns:
            Document: 文档
        """
        return cls(id='0:' + str(uuid.uuid4()),
                   name=os.path.basename(parser.file_path).split('.')[0],
                   file_path=parser.file_path,
                   bookmarks=parser.get_bookmarks(),
                   parser=parser)

    def dump(self, path: str) -> None:
        """ 序列化 Document 对象

        Args:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ 自定义序列化方法
        """
        state = self.__dict__.copy()
        # 移除 parser 属性
        del state['parser']
        return state

    @staticmethod
    def load(path: str, parser: 'Parser') -> 'Document':
        """ 反序列化 Document 对象，不包含parser属性

        Args:
            path: 文件路径
        """
        with open(path, 'rb') as f:
            document: Document = pickle.load(f)
            document.parser = parser
            return document

    def flatten_bookmarks(self) -> list[BookMark]:
        """ 将 bookmark 的树状结构扁平化，以便快速查找

        Returns:
            list[BookMark]: 书签列表
        """
        res: list[BookMark] = []

        def get_bookmark(bookmarks: list[BookMark]):
            for bookmark in bookmarks:
                res.append(bookmark)
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    get_bookmark(bookmark.subs)

        get_bookmark(self.bookmarks)
        return res

    def set_knowledgepoints_by_llm(
        self,
        llm: LLM,
        prompt: ExtractPrompt = ExamplePrompt(),
        self_consistency: bool = False,
        samples: int = 5,
        top: float = 0.5,
        config: Config = Config()) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            prompt (Prompt, optional): 使用的提示词类. Defaults to ExamplePrompt().
            self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
            samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采用自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
            config (Config, optional): 配置. Defaults to Config().
        """

        def get_knowledgepoints(content: str,
                                self_consistency=False,
                                samples: int = 5,
                                top: float = 0.5) -> list[KPEntity]:
            """ 使用 llm 生成知识点实体和与其相关的关系关系、属性

            Args:
                content (str): 输入文本
                self_consistency (bool, optional): 是否采用自我一致性策略. Defaults to False.
                samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
                top (float, optional): 采样自我一致性策略时，出现次数超过 top * samples 时才会被采纳. Defaults to 0.5.

            Returns:
                list[KPEntity]: 生成的知识点实体
            """
            # 实体抽取
            if not self_consistency:
                # 默认策略：生成数量过多则重试，否则随机选择5个
                retry = 0
                while True:
                    prompt_, instruction = prompt.get_ner_prompt(content)
                    llm.instruction = instruction
                    resp = llm.chat(prompt_)
                    entities_name = prompt.post_process(resp)
                    if len(entities_name) <= 8 or retry >= 3:
                        break
                    retry += 1
                if len(entities_name) > 10:
                    entities_name = random.sample(entities_name, 5)
            else:
                # 自我一致性验证
                all_entities_name: list[str] = []
                for idx in range(samples):
                    prompt_, instruction = prompt.get_ner_prompt(content)
                    llm.instruction = instruction
                    resp = llm.chat(prompt_)
                    logger.info(f'第{idx}次采样: ' + resp)
                    entities_name = prompt.post_process(resp)
                    logger.info(f'获取知识点实体: ' + str(entities_name))
                    all_entities_name.extend(entities_name)
                counter = Counter(all_entities_name)
                entities_name = [
                    point for point, count in counter.items()
                    if count > (samples * top)
                ]
            entities: list[KPEntity] = []
            for name in entities_name:
                # 复用知识点实体
                for kp in self.knowledgepoints:
                    if name == kp.name:
                        entities.append(kp)
                        break
                else:
                    kp = KPEntity(id='2:' + str(uuid.uuid4()), name=name)
                    self.knowledgepoints.append(kp)
                    entities.append(kp)
            logger.success(f'最终获取知识点实体: ' + str(entities_name))

            # 属性抽取
            if len(entities_name) == 0:
                pass
            else:
                prompt_, instruction = prompt.get_ae_prompt(
                    content, entities_name)
                llm.instruction = instruction
                resp = llm.chat(prompt_)
                attrs = prompt.post_process(resp) if len(
                    entities_name) != 0 else {}
                logger.success(f'获取知识点属性: ' + str(attrs))

                for name, attr in attrs.items():
                    # 在实体列表中找到名称匹配的实体
                    entity = next((e for e in entities if e.name == name),
                                  None)
                    if entity:
                        # 设置相应的属性值
                        for attr_name, value in attr.items():
                            if attr_name not in entity.attributes:
                                entity.attributes[attr_name] = [value]
                            else:
                                entity.attributes[attr_name].append(value)

            # 关系抽取
            if len(entities_name) <= 1:
                pass
            else:
                prompt_, instruction = prompt.get_re_prompt(
                    content, entities_name)
                llm.instruction = instruction
                resp = llm.chat(prompt_)
                relations = prompt.post_process(resp)
                logger.success(f'获取关系三元组: ' + str(relations))

                for rela in relations:  # list[{'head':'', 'relation':'', 'tail':''}]
                    head, tail = None, None
                    for entity in entities:
                        if entity.name == rela.get('head', None):
                            head = entity
                            break
                    for entity in entities:
                        if entity.name == rela.get('tail', None):
                            tail = entity
                            break
                    if head and tail:
                        for relation in head.relations:
                            if relation.type == rela.get(
                                    'relation', None
                            ) and relation.tail.name == tail.name:  # 确保没有重复的关系
                                break
                        else:
                            head.relations.append(
                                KPRelation(id='3:' + str(uuid.uuid4()),
                                           type=rela['relation'],
                                           tail=tail))

            return entities

        for bookmark in self.flatten_bookmarks():
            if not bookmark.subs:  # 表示最后一级书签 subs为空数组需要设置知识点
                logger.info('子章节: ' + bookmark.title)
                if bookmark.title in config.ignore_page:
                    continue
                contents = self.parser.get_contents(bookmark)
                text_contents = '\n'.join(
                    [content.content for content in contents])
                # 防止生成全空白
                text_contents = text_contents.strip()
                if len(text_contents) == 0:
                    bookmark.subs = []
                    continue
                logger.success('子章节内容: \n' + text_contents)
                entities: list[KPEntity] = get_knowledgepoints(
                    text_contents,
                    self_consistency=self_consistency,
                    samples=samples,
                    top=top)
                bookmark.subs = entities

        # 选择最好的属性值
        for entity in self.knowledgepoints:
            for attr, value_list in entity.attributes.items():
                if len(value_list) == 1:
                    entity.attributes[attr] = value_list[0]
                else:
                    prompt_, instruction = prompt.get_best_attr_prompt(
                        entity.name, attr, value_list)
                    llm.instruction = instruction
                    resp = llm.chat(prompt_)
                    try:
                        idx = int(resp.strip())  # 防止前后空格
                        entity.attributes[attr] = value_list[idx]
                    except ValueError:  # 解析失败，模型回答非纯数字，则随机选择
                        logger.error(resp)
                        entity.attributes[attr] = random.choice(value_list)
                logger.success(
                    f'实体: {entity.name}, 属性: {attr}, 值: {entity.attributes[attr]}'
                )

        # 实体共指消解

    def to_cyphers(self) -> list[str]:
        """ 将整体的关联关系转换为 cypher CREATE 语句

        Returns:
            list[str]: 多条 cypher 语句
        """
        cyphers = [
            f'CREATE (:Document {{id: "{self.id}", name: "{self.name}"}})'
        ]
        # 创建所有知识点实体和实体属性
        for entity in self.knowledgepoints:
            res = [str(sl) for sl in entity.resourceSlices]
            cyphers.append(
                f'CREATE (:KnowledgePoint {{id: "{entity.id}", name: "{entity.name}", define: "{entity.attributes.get("定义", "")}", resource: {res} }})'
            )
        # 创建所有知识点关联
        for entity in self.knowledgepoints:
            for relation in entity.relations:
                if relation.type in ['包含', '相关', '顺序']:
                    cyphers.append(
                        f'MATCH (n1:KnowledgePoint {{id: "{entity.id}"}}) MATCH (n2:KnowledgePoint {{id: "{relation.tail.id}"}}) CREATE (n1)-[:{relation.type} {relation.attributes}]->(n2)'
                    )

        def bookmarks_to_cypher(bookmarks: list[BookMark], parent_id: str):
            cyphers: list[str] = []
            for bookmark in bookmarks:
                # 创建章节实体
                res = [str(resource) for resource in bookmark.resource]
                cyphers.append(
                    f'CREATE (:Chapter {{id: "{bookmark.id}", name: "{bookmark.title}", page_start: {bookmark.page_start.index}, page_end: {bookmark.page_end.index}, resource: {res}}})'
                )
                # 创建章节和上级章节 (书籍) 关联, 所以不写类别
                cyphers.append(
                    f'MATCH (n1 {{id: "{parent_id}"}}) MATCH (n2:Chapter {{id: "{bookmark.id}"}}) CREATE (n1)-[:子章节]->(n2)'
                )
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    cyphers.extend(
                        bookmarks_to_cypher(bookmark.subs, bookmark.id))
                # 创建章节和知识点之间关联
                elif bookmark.subs and isinstance(bookmark.subs[-1], KPEntity):
                    for entity in bookmark.subs:
                        cyphers.append(
                            f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) MATCH (n2:KnowledgePoint {{id: "{entity.id}"}}) CREATE (n1)-[:提到知识点]->(n2)'
                        )
            return cyphers

        cyphers.extend(bookmarks_to_cypher(self.bookmarks, self.id))
        return cyphers

    def set_resource(self, resource_map: 'ResourceMap') -> None:
        """ 为知识点设置相应的资源

        Args:
            resource_map (ResourceMap): 资源映射关系
        """
        title, resource = resource_map.bookmark_title, resource_map.resource
        bookmarks: list[BookMark] = []
        titles = title.split("|")
        for bk in self.flatten_bookmarks():
            if bk.title in titles:
                bookmarks.append(bk)
        if len(bookmarks) > 0:
            for bookmark in bookmarks:
                bookmark.resource.append(resource)
                # 为下面的知识点实体设置Resource Slice
                for kp in bookmark.get_kps():
                    slices = resource.get_slices(kp.name)
                    logger.info(f'{kp.name}: {slices}')
                    if slices:
                        kp.resourceSlices.extend(slices)
