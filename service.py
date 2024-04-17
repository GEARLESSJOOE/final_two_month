import os

from prompt import *
from utils import *
from agent import *

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class Service():
    def __init__(self):
        self.agent = Agent()

    def get_summary_message(self, message, history):
        # 定义用户输入信息总结链
        llm = get_llm_model()
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TPL)
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=os.getenv("VERBOSE")
        )

        # 将用户输入的信息作为聊天历史，并只取最后两条数据
        chat_history = ''
        for question, answer in history[-2:]:
            chat_history += f"问题:{question}, 答案:{answer}\n"
        res = chain.run({"chat_history": chat_history, "question": message})
        return res

    def answer(self, message, history):
        if history:
            message = self.get_summary_message(message, history)
        return self.agent.query(message)



