import os
from glob import glob
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader, PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_embeddings_model


# 定义文本转向量方法
def doc2vec():
    dir_path = os.path.join(os.path.dirname(__file__), "data/inputs/")
    documents = []
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # 文档加载并分割
    for file_path in glob(dir_path + "*.*"):
        loader = None
        if ".csv" in file_path:
            loader = CSVLoader(file_path)  # 只是初始化loader，并返回的是document类型的列表
        if ".pdf" in file_path:
            loader = PyMuPDFLoader(file_path)
        if ".txt" in file_path:
            loader = TextLoader(file_path)
        if loader:
            documents += loader.load_and_split(text_splitter)  # 真正做加载以及分割

    # 向量数据存储
    if documents:
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=get_embeddings_model(),
            persist_directory=os.path.join(os.path.dirname(__file__), "data/db/")
        )
        vector_db.persist()  # 存储向量到Chroma


if __name__ == '__main__':
    doc2vec()
