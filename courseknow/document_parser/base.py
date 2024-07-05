from dataclasses import dataclass, field
from ..llm import LLM, Prompt
import uuid
from loguru import logger
from .config import ignore_page, parser_log
from collections import Counter
from .parser import Parser
import random

logger.remove(0)
logger.add(parser_log,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           mode="w")


@dataclass
class KPEntity:
    id: str
    name: str
    relations: list['KPRelation'] = field(default_factory=list)
    attributes: dict[str, list] | dict[str, str] = field(
        default_factory=dict)  # 同一个属性可能会存在多个属性值, 后续选择一个最好的值


@dataclass
class KPRelation:
    id: str
    type: str
    tail: KPEntity
    attributes: dict[str, list] | dict[str, str] = field(default_factory=dict)


@dataclass
class BookMark:
    """ 书签
    """
    id: str
    title: str
    page_index: int
    page_end: int
    level: int
    subs: list['BookMark'] | list[KPEntity]

    def set_page_end(self, page_end: int) -> None:
        """ 设置书签的结束页, 和直接修改 BookMark 对象的 page_end 属性不同, 该方法会考虑到书签嵌套的情况

        Args:
            page_end (int): 结束页码
        """
        self.page_end = page_end
        if self.subs and isinstance(self.subs[-1], BookMark):
            self.subs[-1].set_page_end(page_end)


@dataclass
class Document:
    """文档
    """
    id: str
    name: str
    bookmarks: list[BookMark]
    parser: Parser
    knowledgepoints: list[KPEntity] = field(default_factory=list)

    def set_knowledgepoints_by_llm(self,
                                   llm: LLM,
                                   prompt: Prompt,
                                   self_consistency: bool = False,
                                   samples: int = 5,
                                   top: float = 0.5) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            prompt (Prompt): 使用的提示词类
            self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
            samples (int, optional): 采样自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采样自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
        """

        def set_knowledgepoints(bookmarks: list[BookMark]) -> None:
            for bookmark in bookmarks:
                if bookmark.title in ignore_page:
                    continue
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    set_knowledgepoints(bookmark.subs)
                else:
                    logger.success('子章节: ' + bookmark.title)
                    contents = self.parser.get_content(bookmark)
                    text_contents = '\n'.join(
                        [content.content for content in contents])
                    # 防止生成全空白
                    text_contents = text_contents.strip()
                    if len(text_contents) == 0:
                        bookmark.subs = []
                        continue
                    entities: list[KPEntity] = get_knowledgepoints(
                        text_contents,
                        self_consistency=self_consistency,
                        samples=samples,
                        top=top)
                    bookmark.subs = entities

        def get_knowledgepoints(content: str,
                                self_consistency=False,
                                samples: int = 5,
                                top: float = 0.5) -> list[KPEntity]:
            """ 使用 llm 生成知识点实体和与其相关的关系关系、属性

            Args:
                content (str): 输入文本
                self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
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
                    resp = llm.chat(prompt.get_ner_prompt(content))
                    entities_name: list[str] = prompt.post_process(resp)
                    if len(entities_name) <= 8 or retry >= 3:
                        break
                    retry += 1
                if len(entities_name) > 10:
                    entities_name = random.sample(entities_name, 5)
            else:
                # 自我一致性验证
                all_entities_name: list[str] = []
                for idx in range(samples):
                    resp = llm.chat(prompt.get_ner_prompt(content))
                    logger.info(f'第{idx}次采样: ' + resp)
                    entities_name = prompt.post_process(resp)
                    if isinstance(entities_name, dict):
                        continue
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
            logger.info(f'最终获取知识点实体: ' + str(entities_name))
            # 属性抽取
            attrs = prompt.post_process(
                llm.chat(prompt.get_ae_prompt(
                    content, entities_name)))  # {'entity': {'attr': ''}}
            logger.info(f'获取知识点属性: ' + str(attrs))
            if not isinstance(attrs, dict):
                pass
            else:
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
            # 关系抽取有强烈的NULL倾向 关系也可以做SC
            if len(entities_name) <= 1:
                pass
            else:
                relations = prompt.post_process(
                    llm.chat(prompt.get_re_prompt(content, entities_name)))
                logger.info(f'获取关系三元组: ' + str(relations))
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

        set_knowledgepoints(self.bookmarks)

        # 选择最好的属性值
        for entity in self.knowledgepoints:
            for attr, value_list in entity.attributes.items():
                if len(value_list) == 1:
                    entity.attributes[attr] = value_list[0]
                else:
                    resp = prompt.get_best_attr(entity.name, attr,
                                                value_list)
                    try:
                        idx = int(resp)
                        entity.attributes[attr] = value_list[idx]
                    except ValueError:  # 解析失败，模型回答非纯数字，则随机选择
                        logger.error(resp)
                        entity.attributes[attr] = random.choice(value_list)
                logger.info(
                    f'实体: {entity.name} 属性: {attr} 值: {entity.attributes[attr]}'
                )

    def get_cyphers(self) -> list[str]:
        """ 将整体的关联关系存入图数据库中

        Returns:
            list[str]: 多条 cypher 语句
        """

        cyphers = [
            f'CREATE (:Document {{id: "{self.id}", name: "{self.name}"}})'
        ]
        # 创建所有知识点实体和实体属性
        for entity in self.knowledgepoints:
            cyphers.append(
                f'CREATE (:KnowledgePoint {{id: "{entity.id}", name: "{entity.name}", define: "{entity.attributes.get("定义", "")}" }})'
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
                if bookmark.title in ignore_page:
                    continue
                # 创建章节实体
                cyphers.append(
                    f'CREATE (:Chapter {{id: "{bookmark.id}", name: "{bookmark.title}", page_start: {bookmark.page_index}, page_end: {bookmark.page_end}}})'
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
