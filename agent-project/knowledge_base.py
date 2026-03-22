"""
本地知识库 - 基于 Chroma + Embedding
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding 模型 (本地运行，轻量)
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

class KnowledgeBase:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self._load_or_create()

    def _load_or_create(self):
        """加载已有知识库或创建新的"""
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"已加载知识库: {self.vectorstore._collection.count()} 条文档")
        else:
            print("知识库为空，请先添加文档")

    def add_documents(self, texts, metadatas=None):
        """添加文档到知识库"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                metadatas=metadatas
            )
        else:
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"已添加 {len(texts)} 条文档")

    def load_folder(self, folder_path, glob="**/*.txt"):
        """从文件夹加载文档"""
        loader = DirectoryLoader(folder_path, glob=glob, loader_cls=TextLoader)
        docs = loader.load()

        # 分割文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = splitter.split_documents(docs)

        # 添加到知识库
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"已加载 {len(splits)} 个文档片段")

    def search(self, query, k=3):
        """搜索知识库"""
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


if __name__ == "__main__":
    # 测试
    kb = KnowledgeBase()

    # 添加一些测试文档
    test_docs = [
        "我的名字叫张三，今年25岁，是一名软件工程师。",
        "我喜欢编程和看科幻电影，周末经常去爬山。",
        "我住在上海浦东新区，工作地点在张江高科技园区。",
    ]
    kb.add_documents(test_docs)

    # 测试搜索
    print("\n搜索结果:")
    results = kb.search("我是谁")
    for r in results:
        print(f"- {r}")

    results = kb.search("我住在哪里")
    for r in results:
        print(f"- {r}")