from py2neo import Graph

from config import *
import re
import json
import itertools
import Levenshtein

"""
该模块主要实现用户问题与图数据库之间的匹配查询，整体步骤分以下几点
1、首先解析保存的实体，进行命名实体识别
2、将识别到的实体与模版词槽进行匹配
2.1、检查实体词槽与模版词槽的关系
2.2、将所有匹配词槽的可能性列举
2.3、将列举后的词槽，进行相似度排序
3、将排序靠前的模版进行解析，并查询图数据库中的信息，返回
"""

class GraphQA():
    # 连接neo4j
    def __init__(self):
        self.graph = Graph(GRAPH_LINK, auth=(AUTH_USERNAME, AUTH_PASSWORD))

    # 整体查询逻辑
    def query(self, text):
        slots = self.get_mention_slots(text)
        template = self.expand_template(slots)
        final_template = self.compute_question_similarity(template, text)
        answer = self.query_and_replace_answer(final_template)
        return answer if answer else "抱歉，没有找到相关消息"

    # 将最终整理好的模版替换最终的res结果并输出
    def query_and_replace_answer(self, templates):
        # print(templates)
        final_answer = []
        for template in templates:
            # print(template)
            try:
                res = self.graph.run(template["cypher"]).data()[0]
                # print(res)
                if res["res"]:
                    final_answer.append(self.replace_token_in_string(template["answer"], res))
            except:
                pass
        return final_answer

    # 计算所有输出模版中的问题与输入问题的相似度，并进行排序
    def compute_question_similarity(self, templates, text):
        for i, template in enumerate(templates):
            # 利用Levenshtein计算文本相似度
            score = Levenshtein.ratio(template["question"], text)
            if score > THRESHOLD:
                template["score"] = score
            else:
                del templates[i]
        res = sorted(templates, key=lambda x: x["score"], reverse=True)  # 默认从小到大(升序)
        return res

    # 定义扩充模版方法，用于模版问题相似度排序
    # 目标格式：[{'%TENT%': 'xxx'}, {"%WENT%": "xxx"}, {"%REL%": "xxx"}]
    def expand_template(self, slots):
        # slots: {'%TENT%': [], '%WENT%': ['武汉斯泰德建设科技有限公司', '武汉江腾铁路工程有限责任公司'], '%REL%': ['Get']}
        # print(TEMPLATES)
        final_template = []
        for template in TEMPLATES:
            cypher_slots = json.loads(template["slots"])  # cypher_slots:{'%WENT%': 1. "%REL%": 1}
            # 判断命名实体识别的词槽与模版定义的词槽数量，如果比定义的词槽数量少，则直接丢弃
            if self.check_slots(cypher_slots, slots):
                # 如果命名实体识别的词槽大于模版定义的词槽数量，则把所有的可能性进行组合
                combinations = self.get_slots_combination(cypher_slots, slots)
                # 模版替换
                for combination in combinations:
                    question = self.replace_token_in_string(template["question"], combination)
                    cypher = self.replace_token_in_string(template["cypher"], combination)
                    answer = self.replace_token_in_string(template["answer"], combination)
                    final_template.append({"question": question, "cypher": cypher, "answer": answer})
        return final_template

    # 模版替换方法
    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    # 整合所有可能性词槽
    def get_slots_combination(self, cypher_slots, slots):
        # 实现元素排列
        combination_value = []
        for key, value in cypher_slots.items():
            res = [*itertools.permutations(slots[key], value)]
            combination_value.append(res)
        # 实现笛卡尔乘积， 目的是将所有可能性的元素排列，与关系进行组合
        combination_lst = []
        for item in [*itertools.product(*combination_value)]:
            dct = {}
            # 将组合的关系，与模版词槽一一对应，并整理成目标格式，以列表形式返回
            for value, key in zip(item, cypher_slots):
                dct[key] = value[0]
            combination_lst.append(dct)
        return combination_lst

    # 检查词槽
    def check_slots(self, cypher_slots, slots):
        for key, value in cypher_slots.items():
            if value > len(slots[key]):
                return False
        return True

    # 命名实体识别整合
    def get_mention_slots(self, text):
        tender_entities = self.parse_tender_entity(text)
        winner_entities = self.parse_winner_entity(text)
        relation = self.parse_relation(text)
        return {
            "%TENT%": tender_entities,
            "%WENT%": winner_entities,
            "%REL%": relation
        }

    # 解析tender实体内容
    def parse_tender_entity(self, text):
        # 拿出所有tender_entity的实体内容
        with open(TENDER_ENTITY_PATH, 'r', encoding="utf-8") as file:
            tender_entity_lst = file.read().split("\n")
        entities = re.findall("|".join(tender_entity_lst), text)
        return entities

    # 解析winner实体内容
    def parse_winner_entity(self, text):
        # 拿出所有winner_entity的实体内容
        with open(WINNER_ENTITY_PATH, "r", encoding="utf-8") as file:
            winner_entity_lst = file.read().split("\n")
        entities = re.findall("|".join(winner_entity_lst), text)
        return entities

    # 解析关系
    def parse_relation(self, text):
        relations = re.findall("|".join(SYNONYMS_MAP["relation"].keys()), text)
        res = list(set([SYNONYMS_MAP["relation"][relation] for relation in relations]))
        return res
