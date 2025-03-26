# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/parser/document.py
# Description: 定义文档以及抽取知识图谱相关方法

from ..llm import LLM, ONTOLOGY
from ..llm.prompt import PromptGenerator, ExamplePromptGenerator, post_process
import shortuuid
from loguru import logger
from collections import Counter
from typing import TYPE_CHECKING, Union
import random
import pickle
import os
from .config import CONFIG
from .utils import instance_method_transactional
from ..resource import ResourceMap
from .types import BookMark, KPEntity, KPRelation, ContentType
from tqdm import tqdm
from course_graph._core import merge
from ..database import Neo4j
from py2neo import Node, Relationship

if TYPE_CHECKING:
    from .parser import Parser


class Document:
    def __init__(self, parser: 'Parser') -> None:
        """ 文档
        
        Args:
            parser (Parser): 对应的解析器
        """
        self.id = '0:' + str(shortuuid.uuid())
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
            parser (Parser): 对应的解析器
            path (str): 文件路径
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

        def get_bookmark(node: Union[Document, BookMark]):
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
            prompt: PromptGenerator = ExamplePromptGenerator(),
            self_consistency: bool = False,
            samples: int = 5,
            top: float = 0.5,
            text_length: int = 400,
            checkpoint: bool = False) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            prompt (Prompt, optional): 使用的提示词类. Defaults to ExamplePromptGenerator().
            self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
            samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采用自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
            text_length (int, optional): 合并文本的长度. Defaults to 400.
            checkpoint (bool, optional): 如果保存有断点信息, 是否继续从断点处运行. Defaults to False.
        """

        @instance_method_transactional('knowledgepoints')
        def get_knowledgepoints(self: 'Document',  # 这里需要传递self的原因是instance_method_transactional本来只能装饰实例方法
                                content: str) -> list[KPEntity]:
            # 实体抽取
            message, instruction = prompt.get_ner_prompt(content)
            llm.instruction = instruction
            if not self_consistency:
                # 默认策略：实体生成数量过多则重试，否则随机选择5个
                retry = 0
                while True:
                    resp, _ = llm.chat(message)
                    entities: dict = post_process(resp) or {}
                    if all(len(value) < 8 for value in entities.values()) or retry >= 3:
                        break
                    retry += 1
                for entity_type, entity_list in entities.items():
                    if len(entity_list) > 10:
                        entities[entity_type] = random.sample(entity_list, 5)
            else:
                # 自我一致性验证
                all_entities: list[dict] = []
                for idx in range(samples):
                    resp, _ = llm.chat(message)
                    logger.info(f'第{idx}次采样: ' + resp)
                    entities: dict = post_process(resp) or {}

                    logger.info(f'获取知识点实体: ' + str(entities))
                    all_entities.append(entities)

                entities = {}
                # 这里的自我一致性是要求每种类型中提及的实体超过一定数量
                for entity_type in {k for d in all_entities for k in d}:  # 所有的 keys
                    elements = [item for d in all_entities if entity_type in d for item in d[entity_type]]
                    entities[entity_type] = [point for point, count in Counter(elements).items() if
                                             count > (samples * top)]
            logger.success(f'最终获取知识点实体: ' + str(entities))
            # 分支结束得到 entities {'entity_type': ['entity1', 'entity2']}

            center_kps: list[KPEntity] = []  # only for center entity
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:  # entity_type 不再作为单独出现而是作为属性
                    # 复用知识点实体
                    if kp := next((kp for kp in self.knowledgepoints if kp.name == entity_name), None):
                        kp.marginalized = False
                        center_kps.append(kp)
                    else:
                        kp = KPEntity(id='2:' + str(shortuuid.uuid()), name=entity_name, type=entity_type)
                        self.knowledgepoints.append(kp)
                        center_kps.append(kp)

            # 属性抽取
            if len(center_kps) == 0:
                pass
            else:
                message, instruction = prompt.get_ae_prompt(content, [kp.name for kp in center_kps])  # 只使用 name
                llm.instruction = instruction
                resp, _ = llm.chat(message)
                attrs: dict = post_process(resp) or {}
                logger.success(f'获取知识点属性: ' + str(attrs))

                for name, attr in attrs.items():
                    # 使用 name 匹配
                    if matching_kp := next((kp for kp in center_kps if kp.name == name), None):
                        # 更新相应的属性值
                        if isinstance(attr, dict):
                            for attr_name, value in attr.items():
                                matching_kp.cached_attributes.setdefault(attr_name, []).append(value)

            message, instruction = prompt.get_re_prompt(content, [kp.name for kp in center_kps])  # 只使用 name
            llm.instruction = instruction
            if not self_consistency:
                resp, _ = llm.chat(message)
                relations = post_process(resp) or []
            else:
                all_relations = []
                for idx in range(samples):
                    resp, _ = llm.chat(message)
                    logger.info(f'第{idx}次采样: ' + resp)
                    relations = post_process(resp) or []

                    logger.info(f'获取知识点实体: ' + str(entities))
                    all_relations.extend(relations)
                relations = [
                    dict(relation)
                    for relation, count in Counter(frozenset(relation.items()) for relation in all_relations).items() if
                    count > (samples * top)
                ]

            logger.success(f'最终获取关系三元组: ' + str(relations))
            # 分支结束得到 relations [{'head':'', 'relation':'', 'tail':''}]

            for rela in relations:
                head, tail = None, None
                if head_name := rela.get('head'):
                    kp = next((kp for kp in center_kps if kp.name == head_name), None)
                    if not kp:
                        kp = KPEntity(id='2:' + str(shortuuid.uuid()), name=head_name, type='', marginalized=True)
                        self.knowledgepoints.append(kp)
                        head = kp
                if tail_name := rela.get('tail'):
                    kp = next((kp for kp in center_kps if kp.name == tail_name), None)
                    if not kp:
                        kp = KPEntity(id='2:' + str(shortuuid.uuid()), name=tail_name, type='', marginalized=True)
                        self.knowledgepoints.append(kp)
                        tail = kp
                if head and tail:
                    for relation in head.relations:
                        if relation.type == rela.get('relation') and relation.tail.name == tail.name:  # 确保没有重复的关系
                            break
                        else:
                            head.relations.append(
                                KPRelation(id='3:' + str(shortuuid.uuid()),
                                            type=rela['relation'],
                                            tail=tail))
            return center_kps

        # 知识抽取
        for index, bookmark in tqdm(enumerate(self.flatten_bookmarks()), total=len(self.flatten_bookmarks()),
                                    desc='知识抽取'):
            if not bookmark.subs:  # 表示最后一级书签 subs为空数组需要设置知识点
                logger.info('子章节: ' + bookmark.title)
                if index < self.checkpoint['extract_index'] and checkpoint:
                    logger.info('已跳过')
                    continue
                if bookmark.title in CONFIG['IGNORE_PAGE']:
                    logger.info('已跳过')
                    continue
                contents = self.parser.get_contents(bookmark)
                kps: list[KPEntity] = []
                texts = []
                for idx, content in enumerate(contents):
                    texts.append(content.content)
                    if content.type == ContentType.Title:
                        texts[-1] += '\n'
                    elif idx != len(contents) - 1 and contents[idx + 1].type == ContentType.Title:
                        texts[-1] += '\n'
                contents = merge(texts, n=text_length)  # 优化换行位置
                for content in contents:
                    if len(content) != 0:
                        logger.info('输入片段: \n' + content)
                        try:
                            entities: list[KPEntity] = get_knowledgepoints(
                                self,
                                content)
                        except Exception as e:
                            raise e
                        else:
                            kps.extend(entities)
                self.checkpoint['extract_index'] = index
                bookmark.subs = list({kp.id: kp for kp in kps}.values())  # 去重

        # 属性值总结
        for entity in tqdm(self.knowledgepoints, desc='属性总结'):
            for attr, value_list in entity.cached_attributes.items():
                if len(value_list) == 1:
                    entity.attributes[attr] = value_list[0]
                else:
                    prompt_, instruction = prompt.get_best_attr_prompt(
                        entity.name, attr, value_list)
                    llm.instruction = instruction
                    resp, _ = llm.chat(prompt_)
                    entity.attributes[attr] = resp
                logger.success(
                    f'实体: {entity.name}, 属性: {attr}, 值: {entity.cached_attributes[attr]}'
                )

        # A.对边缘化的知识点进行处理 存放在 self.knowledgepoints 中但不属于层级中
        # B.共指消解

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

    def to_graph(self, neo4j: Neo4j) -> None:
        """ 将 document 对象转换为图谱
        
        Args:
            neo4j (Neo4j): Neo4j 对象
        
        """
        relas = list(ONTOLOGY['relations'].keys())
        attrs = list(ONTOLOGY['attributes'].keys())
        transaction = neo4j.begin()
        id2node = {}

        # 创建文档实体
        document_node = Node('文档',
                             id=self.id,
                             name=self.name)
        transaction.create(document_node)

        # 创建所有知识点实体和实体属性
        for entity in self.knowledgepoints:
            res = [str(sl) for sl in entity.resourceSlices]
            node = Node('知识点',
                        id=entity.id,
                        name=entity.name,
                        type=entity.type,
                        resource=res,
                        **{attr: entity.attributes.get(attr, '') for attr in attrs})
            transaction.create(node)
            id2node[entity.id] = node

        # 创建所有知识点关联
        for entity in self.knowledgepoints:
            for relation in entity.relations:
                if relation.type in relas:
                    transaction.create(Relationship(id2node[entity.id],
                                                    relation.type,
                                                    id2node[relation.tail.id],
                                                    id=relation.id))

        def bookmark_to_graph(bookmark: BookMark, parent_node: Node):
            if bookmark.title in CONFIG['IGNORE_PAGE']:
                return
            # 创建章节实体
            bookmark_node = Node('章节',
                                 id=bookmark.id,
                                 name=bookmark.title,
                                 page_start=bookmark.page_start.index,
                                 page_end=bookmark.page_end.index,
                                 resource=[str(resource) for resource in bookmark.resource])
            transaction.create(bookmark_node)
            # 创建章节和上级章节/文档关联
            transaction.create(Relationship(parent_node,
                                            '子章节',
                                            bookmark_node,
                                            id=f'3:{shortuuid.uuid()}'))
            # 创建章节和下级章节/知识点关联
            for sub in bookmark.subs:
                match sub:
                    case BookMark():
                        bookmark_to_graph(sub, bookmark_node)
                    case KPEntity():
                        transaction.create(Relationship(bookmark_node,
                                                        '包含知识点',
                                                        id2node[sub.id],
                                                        id=f'3:{shortuuid.uuid()}'))

        for _, bookmark in enumerate(self.bookmarks):
            bookmark_to_graph(bookmark, document_node)
        transaction.commit()

    def to_json(self) -> tuple[list, list]:
        """ 将 document 对象转换为 json 格式

        Returns:
            tuple[list, list]: relations, entity_attributes 两个字段
        """
        relations = []
        entity_attributes = []

        def add_relation(x_id, x_type, x_name, y_id, y_type, y_name, relation, relation_id=None):
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
                        if bookmark.title in CONFIG['IGNORE_PAGE']:
                            pass
                        add_relation(node.id, '文档', node.name, bookmark.id, '章节', bookmark.title, '子章节')
                        dfs(bookmark)
                case BookMark():
                    for sub in node.subs:
                        match sub:
                            case BookMark():
                                add_relation(node.id, '章节', node.title, sub.id, '章节', sub.title, '子章节')
                            case KPEntity():
                                add_relation(node.id, '章节', node.title, sub.id, '知识点', sub.name, '提到知识点')
                        dfs(sub)
                case KPEntity():
                    for relation in node.relations:
                        add_relation(node.id, '知识点', node.name, relation.tail.id, '知识点', relation.tail.name,
                                     relation.type, relation.id)

        dfs(self)

        for kp in self.knowledgepoints:
            attribute = {
                'entity_id': kp.id,
                'entity_type': '知识点/' + kp.type,
                'entity_name': kp.name,
            }
            for attr in list(ONTOLOGY['attributes'].keys()):
                attribute[attr] = kp.attributes.get(attr, '')
            entity_attributes.append(attribute)

        return relations, entity_attributes  # , relation_attributes
