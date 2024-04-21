import json

from config import *
from py2neo import Graph
from collections import defaultdict

"""
利用爬虫数据，构建知识图谱
1. 解析爬取到的json文件中的数据
2. 创建entity_data,以及relation_data 用于保存节点以及关系
2.1 entity_data保存Tender，以及Winner表，以及表中的属性
2.2 relation_data保存两个表之间的属性的关系
3. 利用构建好的数据，创建知识图谱（create 以及 unwind）
"""


class BuildGraph():
    def __init__(self):
        # 定义表
        self.entity_data = defaultdict(list)
        self.relation_data = defaultdict(list)
        # 按行解析crawler的json文件
        self.parse_crawler_data()
        # 初始化Graph
        self.graph = Graph(GRAPH_LINK, auth=(AUTH_USERNAME, AUTH_PASSWORD))

    def parse_crawler_data(self):
        with open(CRAWLER_PATH, "r", encoding="utf-8") as file:
            tender_data = file.readlines()  # 按行读取json数据(list)
            for line in tender_data:  # 此时line类型为str
                row_data = json.loads(line)  # 将str类型转换为json中的原始数据类型即字典

                # 添加Tender表中的属性
                self.entity_data["Tender"].append({
                    "tender_name": row_data["tender_name"],
                    "tender_type": row_data["tender_type"],
                    "tender_info": row_data["tender_intro"],
                    "tenderee": row_data["tenderee"]
                })

                # 添加Winner表中的属性
                self.entity_data["Winner"].append({
                    "winning_company": row_data["winning_company"],
                    "winning_price": row_data["winning_price"]
                })

                # 创建节点关系
                self.relation_data["Get"].append((row_data["winning_company"], row_data["tender_name"]))

    # 创建实体节点
    # create(:Tender{name:"xxx", type:"xxx", info:"xxx"})
    def create_entity(self):
        for label, entity_lst in self.entity_data.items():
            for entity in entity_lst:
                # 字符串拼接，也可以考虑用格式化方法
                text = ','.join([k + ':"' + v.replace("【", "[").replace("】", "]") + '"' for k, v in entity.items()])
                cql = "create(:" + label + "{" + text + "})"
                self.graph.run(cql)

    # 创建实体关系
    # unwind[{from: "xxx", to: "xxx"}, {from: "xxx", to: "xxx"}] as res
    # match(n1{winning_company: res.from}), (n2{tender_name: res.to})
    # merge(n1)-[:Get]->(n2)
    def create_relation(self):
        for label, relation_lst in self.relation_data.items():
            # 字符串拼接
            text = ",".join(
                ['{from: "' + k + '", to: "' + v.replace("【", "[").replace("】", "]") + '"}' for k, v in relation_lst])
            cql = f"""
            unwind[{text}] as res
            match(n1{{winning_company: res.from}})
            match(n2{{tender_name: res.to}})
            merge(n1)-[:{label}]->(n2)
            """
            self.graph.run(cql)


if __name__ == '__main__':
    bg = BuildGraph()
    # bg.create_entity()
    bg.create_relation()
