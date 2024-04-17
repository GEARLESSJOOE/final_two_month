import os.path

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from config import *
from prompt import *
from utils import *

from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.llm_requests import LLMRequestsChain
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory


class Agent():
    def __init__(self):
        # 加载向量数据库中数据
        self.vector_db = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), "data/db"),
            embedding_function=get_embeddings_model()
        )

    # 利用大模型自身能力通用回答
    def generic_func(self, question):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv("VERBOSE")
        )
        return chain.run({"question": question})

    # 向量数据库检索回答
    def retrieval_func(self, question):
        # 召回向量数据库中最相关的5条数据
        documents = self.vector_db.similarity_search_with_relevance_scores(question, k=5)
        # 对召回的数据利用分数再一次筛选
        retrieval_results = [doc[0].page_content for doc in documents if doc[1] > 0.7]
        prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT_TPL)
        chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv("VERBOSE")
        )
        res = chain.run({"retrieval_results": retrieval_results, "question": question})
        return res

    # Neo4j数据库知识图谱查询（待完善）
    def graph_func(self, question):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type="list", name="policy", description="国家政策实体"),
            ResponseSchema(type="list", name="commodity", description="商品名称实体"),
            ResponseSchema(type="list", name="company", description="公司名称实体")
        ]
        # 格式化模版
        format_instructions = structured_output_parser(response_schemas=response_schemas)

        # 设置解析器
        output_parser = StructuredOutputParser(response_schemas=response_schemas)

        ner_prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={"format_instructions": format_instructions},
            input_variables=["question"]
        )
        chain = LLMChain(
            llm=get_llm_model(),
            prompt=ner_prompt,
            verbose=os.getenv("VERBOSE")
        )
        res = chain.run({"question": question})
        ner_res = output_parser.parse(res)
        return ner_res

    # 利用google进行搜索查询
    def search_func(self, question):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv("VERBOSE")
        )
        request_chain = LLMRequestsChain(
            llm_chain=llm_chain,
            requests_key="requests_result"
        )
        inputs = {
            "question": question,
            "url": "https://www.google.com.hk/search?q=" + question.replace(" ", "+")
        }
        res = request_chain.run(inputs)
        return res

    # 定义agent模块
    def query(self, question):
        # 定义tools
        tools = [
            Tool(
                name="generic_func",
                func=self.generic_func,
                description="当用户输入日常聊天词汇时，比如，你是谁，你都有什么功能，你能为我做些什么，你都可以提供那些帮助等问题时，你应该调用该工具"
            ),
            Tool(
                name="retrieval_func",
                func=self.retrieval_func,
                description="当用户输入招标领域内的专业问题时，比如，供应商异议驳回，政府采购方式，招标通知书效力等问题时，你应该调用该工具"
            ),
            Tool(
                name="search_func",
                func=self.search_func,
                description="其他工具没有正确答案时，通过搜索引擎，回答通用类问题"
            )
        ]

        # 定义agent
        prefix = """请用中文尽你所能回答下列问题，可以使用以下工具"""
        suffix = """开始
        History: {chat_history}
        Question: {human_input}
        Thought: {agent_scratchpad}
        """
        agent_prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["chat_history", "human_input", "agent_scratchpad"]
        )
        # 第一个chain用来agent预测
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=agent_prompt,
            verbose=os.getenv("VERBOSE")
        )
        agent = ZeroShotAgent(llm_chain=llm_chain)

        # 定义memory组件
        memory = ConversationBufferMemory(memory_key="chat_history")

        # 定义AgentExecutor，第二个chain用来agent执行
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=os.getenv("VERBOSE"),
            handle_parsing_errors=True
        )
        res = agent_chain.run({"human_input": question})
        return res



