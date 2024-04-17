from langchain_openai import OpenAIEmbeddings  # 更改最新langchain框架中的倒包路径，直接使用浪chain_openai导入
from langchain_openai import ChatOpenAI
from py2neo import Graph
from config import *

import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings_model():
    model_map = {
        'openai': OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDINGS_MODEL')
        )
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))


def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model=os.getenv('OPENAI_LLM_MODEL'),
            temperature=os.getenv('TEMPERATURE'),
            max_tokens=os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))


def structured_output_parser(response_schemas):
    text = """
    1. 请从以下文本，抽取出实体信息，并按照json格式输出，json包含首尾的"'''json"和"'''"
    2. 注意：根据用户输入的事实抽取内容，不要推理，不要补充信息，句子中如果不存在该实体，就输出空字段
    3. 以下是字段含义和类型，要求输出的json中，必须包含下列所有字段 \n
    """
    for schema in response_schemas:
        text += schema.name + "字段, 表示" + schema.description + ",类型为" + schema.type + "\n"
    return text


