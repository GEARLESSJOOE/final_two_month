import os.path

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.documents import Document

from question_match import GraphQA

from config import *
from prompt import *
from utils import *

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.llm_requests import LLMRequestsChain
from langchain_community.vectorstores import Chroma, FAISS
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain import hub


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
        return chain.invoke({"question": question})["text"]

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
        res = chain.invoke({"retrieval_results": retrieval_results, "question": question})["text"]
        return res

    # Neo4j数据库知识图谱查询
    def graph_func(self, x, question):
        # # 方法1：替换langchain，利用正则匹配进行识别
        # qa = GraphQA()
        # answer = qa.query(question)
        #
        # graph_prompt = PromptTemplate(
        #     template=GRAPH_SUMMARY_TPL,
        #     partial_variables={"full_information": answer},
        #     input_variables=["question"]
        # )
        #
        # chain = LLMChain(
        #     llm=get_llm_model(),
        #     prompt=graph_prompt,
        #     verbose=os.getenv("VERBOSE")
        # )
        #
        # res = chain.invoke({"question": question})['text']
        # return res

        # 方法2：命名实体识别(langchain实现)
        response_schemas = [
            # ResponseSchema(type="list", name="tender_company", description="招标公司实体"),
            ResponseSchema(type="list", name="winning_company", description="中标公司名称实体"),
            ResponseSchema(type="list", name="winning_price", description="中标金额实体")
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
        res = chain.invoke({"question": question})["text"]
        ner_res = output_parser.parse(res)  # 必须按照首'''json尾'''进行查找
        # print(ner_res)

        # 词槽替换
        graph_templates = []
        for template in LANGCHAIN_GRAPH_TEMPLATE:
            slots_key = template["slots"][0]
            for value in ner_res[slots_key]:
                question = replace_token_in_string(template["question"], {slots_key: value})
                cypher = replace_token_in_string(template["cypher"], {slots_key: value})
                answer = replace_token_in_string(template["answer"], {slots_key: value})
                graph_templates.append({"question": question, "cypher": cypher, "answer": answer})
        # print(graph_templates)

        # 相似度检索
        graph_documents = [Document(page_content=template["question"], metadata=template) for template in
                           graph_templates]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(question, k=2)
        # print(graph_documents_filter)

        # 执行CQL,拿最终结果
        question_result = []
        graph = get_connect_neo4j()
        for item in graph_documents_filter:
            question = item[0].page_content
            cypher = item[0].metadata["cypher"]
            answer = item[0].metadata["answer"]
            try:
                result = graph.run(cypher).data()
                if result[0].values():
                    final_answer = replace_token_in_string(answer, result[0])
                    question_result.append(f'问题：{question}，回答：{final_answer}')
                    # print(question_result)
            except:
                pass

        # 将结果交给大模型处理总结
        graph_prompt = PromptTemplate(
            template=GRAPH_SUMMARY_TPL,
            partial_variables={"full_information": question_result},
            input_variables=["question"]
        )

        chain = LLMChain(
            llm=get_llm_model(),
            prompt=graph_prompt,
            verbose=os.getenv("VERBOSE")
        )

        res = chain.invoke({"question": question})['text']
        return res

    # 利用360进行搜索查询
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
            "url": "https://www.so.com/s?q=" + question.replace(" ", "+")
        }
        res = request_chain.invoke(inputs)["output"]
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
                name="graph_func",
                func=lambda x: self.graph_func(x, question),
                description="当用户询问某公司所拥有的招标项目或者某公司得到了哪些公司招标项目时，你应该调用该工具"
            ),
            Tool(
                name="search_func",
                func=self.search_func,
                description="其他工具没有正确答案时，通过搜索引擎，回答通用类问题"
            )
        ]

        # 定义agent
        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = "请用中文回答问题！Final Answer 必须尊重 Observation 的结果，不能改变语意\n\n" + prompt.template
        # 将ZeroShotAgent替换为create_react_agent
        agent = create_react_agent(
            llm=get_llm_model(),
            prompt=prompt,
            tools=tools
        )
        memory = ConversationBufferMemory(memory_key="chat_history")

        # 定义AgentExecutor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            verbose=os.getenv("VERBOSE")
        )

        res = agent_executor.invoke({"input": question})["output"]
        return res

        # # 定义agent
        # prefix = """请用中文尽你所能回答下列问题，可以使用以下工具"""
        # suffix = """开始
        # History: {chat_history}
        # Question: {human_input}
        # Thought: {agent_scratchpad}
        # """
        # agent_prompt = ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["chat_history", "human_input", "agent_scratchpad"]
        # )
        # # 第一个chain用来agent预测
        # llm_chain = LLMChain(
        #     llm=get_llm_model(),
        #     prompt=agent_prompt,
        #     verbose=os.getenv("VERBOSE")
        # )
        # agent = ZeroShotAgent(llm_chain=llm_chain)
        #
        # # 定义memory组件
        # memory = ConversationBufferMemory(memory_key="chat_history")
        #
        # # 定义AgentExecutor，第二个chain用来agent执行
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent=agent,
        #     tools=tools,
        #     memory=memory,
        #     verbose=os.getenv("VERBOSE"),
        #     handle_parsing_errors=True
        # )
        # res = agent_chain.run({"human_input": question})
        # return res


if __name__ == '__main__':
    agent = Agent()
    # print(agent.retrieval_func("供应商针对单一来源异议被驳回，可以再次异议吗？"))
    # print(agent.graph_func("中南建筑设计院股份有限公司赢得了华中农业大学学生宿舍及生活配套设施项目方案设计华中农业大学学生宿舍及生活配套设施项目方案，价格为198989"))
    # agent.graph_func("zhangjunfeng")
    # agent.query("武汉斯泰德建设科技有限公司, 武汉江腾铁路工程有限责任公司得到了哪些公司的项目？")
    agent.query("中南建筑设计院股份有限公司招标项目都有哪些？")
    # print(agent.graph_func(x=[], question="中南建筑设计院股份有限公司，以及武汉斯泰德建设科技有限公司都有什么项目"))
    # print(agent.graph_func(x=[], question="如果我有8000，我能招标什么项目"))
