import os

BASE_PATH = os.path.dirname(__file__)
CRAWLER_PATH = os.path.join(BASE_PATH, "data/crawler/tender_data.json")
TENDER_ENTITY_PATH = os.path.join(BASE_PATH, "data/entity/tender_entity")
WINNER_ENTITY_PATH = os.path.join(BASE_PATH, "data/entity/winner_entity")

GRAPH_LINK = "bolt://localhost:7687"
AUTH_USERNAME = "neo4j"
AUTH_PASSWORD= "zjf568938593"
THRESHOLD = 0.1


SYNONYMS_MAP = {
    "relation": {
        "赢得": "Get",
        "获得": "Get",
        "拿到": "Get",
        "得到": "Get"
    }
}

TEMPLATES = [
    {
        'question': '%WENT%目前%REL%了哪些公司的项目？',
        'cypher': "match (n1 {winning_company:'%WENT%'})-[:%REL%]->(n2) return substring(reduce(s = '', x in collect(n2.tenderee) | s + '/' + x), 1) as res",
        'answer': '%WENT%目前%REL%的项目：res',
        'slots': '{"%WENT%": 1, "%REL%": 1}',
        'example': '中建三局集团有限公司赢得了哪些公司的项目？'
    },
    {
        'question': '%TENT%都有哪些招标项目？',
        'cypher': "match (n1 {tenderee:'%TENT%'}) return substring(reduce(s = '', x in collect(n1.tender_name) | s + '/' + x), 1) as res",
        'answer': '%TENT%获得的招标项目有：res',
        'slots': '{"%TENT%": 1}',
        'example': '武汉天河机场有限责任公司都有哪些招标项目？'
    },
]

LANGCHAIN_GRAPH_TEMPLATE = [
    {
        'question': '%winning_company%有什么项目？',
        'cypher': "match (n1 {winning_company:'%winning_company%'})-[:Get]->(n2) return substring(reduce(s = '', x in collect(n2.tender_name) | s + '/' + x), 1) as res",
        'answer': '%winning_company%的项目有：%res%',
        'slots': ["winning_company"],
        'example': '中建三局集团有限公司有什么项目？'
    },
    {
        'question': '如果我有%winning_price%元，可以获得哪些招标项目？',
        'cypher': "match(n:Winner) where n.winning_price < tointeger(%winning_price%) match (n)-[:Get]->(m) return substring(reduce(s = '', x in collect(m.tender_name) | s + '/' + x), 1) as res",
        'answer': '如果有%winning_price%元可以获得的招标项目有：%res%',
        'slots': ["winning_price"],
        'example': '如果我有4000元，可以获得哪些招标项目？'
    }
]


