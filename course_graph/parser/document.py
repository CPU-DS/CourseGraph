# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/document.py
# Description: 定义文档以及抽取知识图谱相关方法

from ..llm import LLM, ExtractPrompt, ExamplePrompt
from ..llm.prompt import ontology
import shortuuid
from loguru import logger
from collections import Counter
from typing import TYPE_CHECKING, Union
import random
import pickle
import os
from .config import config
from .utils import instance_method_transactional
from ..resource import ResourceMap
from .bookmark import BookMark
from .entity import KPEntity, KPRelation

if TYPE_CHECKING:
    from .parser import Parser


class Document:
    def __init__(self, parser: 'Parser') -> None:
        """ 文档
        
        Args:
            parser (Parser): 对应的解析器
        """
        self.id ='0:' + str(shortuuid.uuid())
        self.name = os.path.basename(parser.file_path).split('.')[0]
        self.parser = parser
        self.file_path = parser.file_path
        self.bookmarks = parser.get_bookmarks()

        self.knowledgepoints: list[KPEntity] = []  # 全局共享状态
        self.checkpoint = {
            'extract_index': 0
        }

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

        def get_bookmark(node:Union[Document, BookMark]):
            match node:
                case Document():
                    for bookmark in node.bookmarks:
                        res.append(bookmark)
                        get_bookmark(bookmark)
                case BookMark():
                    for sub in node.subs:
                        match sub:
                            case BookMark():
                                res.append(sub)
                                get_bookmark(sub)
                            case _:
                                pass
        get_bookmark(self)
        return res

    @logger.catch
    def set_knowledgepoints_by_llm(
            self,
            llm: LLM,
            prompt: ExtractPrompt = ExamplePrompt(),
            self_consistency: bool = False,
            samples: int = 5,
            top: float = 0.5,
            checkpoint: bool = False) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            prompt (Prompt, optional): 使用的提示词类. Defaults to ExamplePrompt().
            self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
            samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采用自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
            checkpoint (bool, optional): 如果保存有断点信息, 是否继续从断点处运行. Defaults to False.
        """

        @instance_method_transactional('knowledgepoints')
        def get_knowledgepoints(self,  # 这里需要传递self的原因是instance_method_transactional本来只能装饰实例方法
                                content: str,
                                self_consistency=False,
                                samples: int = 5,
                                top: float = 0.5) -> list[KPEntity]:
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
                    kp = KPEntity(id='2:' + str(shortuuid.uuid()), name=name)
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
                                KPRelation(id='3:' + str(shortuuid.uuid()),
                                           type=rela['relation'],
                                           tail=tail))

            return entities

        # 知识抽取
        for index, bookmark in enumerate(self.flatten_bookmarks()):
            if not bookmark.subs:  # 表示最后一级书签 subs为空数组需要设置知识点
                logger.info('子章节: ' + bookmark.title)
                if index < self.checkpoint['extract_index'] and checkpoint:
                    logger.info('已跳过')
                    continue
                if bookmark.title in config.ignore_page:
                    logger.info('已跳过')
                    continue
                contents = self.parser.get_contents(bookmark)
                text_contents = '\n'.join(
                    [content.content for content in contents])
                # 防止生成全空白
                text_contents = text_contents.strip()
                if len(text_contents) == 0:
                    bookmark.subs = []
                    continue
                logger.info('子章节内容: \n' + text_contents)
                try:
                    entities: list[KPEntity] = get_knowledgepoints(
                        self,
                        text_contents,
                        self_consistency=self_consistency,
                        samples=samples,
                        top=top)
                except Exception as e:
                    self.checkpoint['extract_index'] = index
                    raise e
                else:
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
        """ 将图谱转换为 cypher CREATE 语句

        Returns:
            list[str]: 多条 cypher 语句
        """
        relas = list(ontology.relations.keys())
        attrs = list(ontology.attributes.keys())
        cyphers = [
            f'CREATE (:Document {{id: "{self.id}", name: "{self.name}"}})'
        ]
        # 创建所有知识点实体和实体属性
        for entity in self.knowledgepoints:
            res = [str(sl) for sl in entity.resourceSlices]
            attr_string = [f'{attr}: "{entity.attributes.get(attr, "")}' for attr in attrs]
            cyphers.append(
                f'CREATE (:KnowledgePoint {{id: "{entity.id}", name: "{entity.name}", resource: {res},  {",".join(attr_string)}}})'
            )
        # 创建所有知识点关联
        for entity in self.knowledgepoints:
            for relation in entity.relations:
                if relation.type in relas:
                    cyphers.append(
                        f'MATCH (n1:KnowledgePoint {{id: "{entity.id}"}}) MATCH (n2:KnowledgePoint {{id: "{relation.tail.id}"}}) CREATE (n1)-[:{relation.type} {relation.attributes}]->(n2)'
                    )

        def bookmark_to_cypher(bookmark: BookMark, parent_id: str):
            if bookmark.title in config.ignore_page:
                return
            # 创建章节实体
            res = [str(resource) for resource in bookmark.resource]
            cyphers.append(
                f'CREATE (:Chapter {{id: "{bookmark.id}", name: "{bookmark.title}", page_start: {bookmark.page_start.index}, page_end: {bookmark.page_end.index}, resource: {res}}})'
            )
            # 创建章节和上级章节 (书籍) 关联, 所以不写类别
            cyphers.append(
                f'MATCH (n1 {{id: "{parent_id}"}}) MATCH (n2:Chapter {{id: "{bookmark.id}"}}) CREATE (n1)-[:子章节]->(n2)'
            )
            for sub in bookmark.subs:
                match sub:
                    case BookMark():
                        bookmark_to_cypher(sub, bookmark.id)
                    case KPEntity():
                        cyphers.append(
                            f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) MATCH (n2:KnowledgePoint {{id: "{entity.id}"}}) CREATE (n1)-[:提到知识点]->(n2)'
                        )

        for bookmark in self.bookmarks:
            bookmark_to_cypher(bookmark, self.id)
        return cyphers

    def to_json(self) -> tuple[dict, dict]:
        """ 将图谱转换为 json 格式

        Returns:
            tuple[dict, dict]: relations, entity_attributes 两个字段
        """
        relations = []
        entity_attributes = []
        # relation_attributes = []

        def add_relation(x_id, x_type, x_name, y_id, y_type ,y_name, relation, relation_id=None):
            relations.append({
                'x_id': x_id,
                'x_type': x_type,
                'x_name': x_name,
                'y_id': y_id,
                'y_type': y_type,
                'y_name': y_name,
                'relation': relation,
                'relation_id': relation_id or f'3:{shortuuid.uuid()}'
            })

        def dfs(node: Union[Document, BookMark, KPEntity]):
            match node:
                case Document():
                    for bookmark in node.bookmarks:
                        if bookmark in config.ignore_page:
                            pass
                        add_relation(node.id, 'Document', node.name, bookmark.id, 'Chapter',bookmark.title, '子章节')
                        dfs(bookmark)
                case BookMark():
                    for sub in node.subs:
                        match sub:
                            case BookMark():
                                add_relation(node.id, 'Chapter', node.title, sub.id, 'Chapter', sub.title, '子章节')
                            case KPEntity():
                                add_relation(node.id, 'Chapter', node.title, sub.id, 'KnowledgePoint', sub.name, '提到知识点')
                        dfs(sub)
                case KPEntity():
                    for relation in node.relations:
                        add_relation(node.id, node.name, 'KnowledgePoint', relation.tail.id, 'KnowledgePoint', relation.tail.name, relation.type, relation.id)

        dfs(self)

        attrs = list(ontology.attributes.keys())
        for kp in self.knowledgepoints:
            attribute = {
                'entity_id': kp.id,
                'entity_type': 'KnowledgePoint',
                'entity_name': kp.name,
            }
            for attr in attrs:
                attribute[attr] = kp.attributes.get(attr, '')
            entity_attributes.append(attribute)

        return relations, entity_attributes #, relation_attributes

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
                    logger.success(f'{kp.name}: {slices}')
                    if slices:
                        kp.resourceSlices.extend(slices)
