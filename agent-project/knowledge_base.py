"""
本地知识库 - 基于 Chroma + Embedding
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用 SiliconFlow 的嵌入模型
# 从 API 获取的可用模型：Qwen/Qwen3-Embedding-0.6B 是最小最快的
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

class KnowledgeBase:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        # 使用 SiliconFlow API 调用嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY")
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

    def search(self, query, k=5):
        """搜索知识库"""
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def learn(self, text, threshold=0.85):
        """
        学习新知识：自动判断是新增还是更新相似内容

        Args:
            text: 新学习的文本
            threshold: 相似度阈值，超过则认为是重复内容（会更新）

        Returns:
            str: 操作结果描述
        """
        if not text or not text.strip():
            return "学习内容为空"

        # 先搜索是否有相似内容
        if self.vectorstore is not None:
            docs = self.vectorstore.similarity_search(text, k=3)
            for doc in docs:
                # 如果有高度相似的内容，更新它
                # 注意：Chroma 的 similarity_search 返回的是文档对象
                if hasattr(doc, 'page_content') and doc.page_content:
                    return self.update_by_content(doc.page_content, text)
            # 没有相似内容，新增
            self.add_documents([text])
            return f"已学习新知识: {text[:50]}{'...' if len(text) > 50 else ''}"
        else:
            # 知识库为空，直接添加
            self.vectorstore = Chroma.from_texts(
                texts=[text],
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            return f"已添加新知识: {text[:50]}{'...' if len(text) > 50 else ''}"

    def update_by_content(self, old_content, new_content):
        """根据旧内容查找并更新"""
        try:
            # 获取所有文档的 metadata 找到旧内容对应的 id
            collection = self.vectorstore._collection
            # Chroma 不直接支持按内容更新，用删除+添加模拟
            # 先获取所有数据找到匹配的
            all_data = collection.get(include=["documents", "metadatas"])
            for i, doc in enumerate(all_data.get("documents", [])):
                if doc == old_content:
                    # 删除旧文档
                    collection.delete(ids=[all_data["ids"][i]])
                    # 添加新文档
                    self.vectorstore.add_texts(texts=[new_content])
                    return f"已更新知识: {old_content[:30]}... -> {new_content[:50]}..."

            # 没找到匹配的，直接添加
            self.add_documents([new_content])
            return f"已添加新知识: {new_content[:50]}{'...' if len(new_content) > 50 else ''}"
        except Exception as e:
            # 更新失败就追加
            self.add_documents([new_content])
            return f"添加失败，已追加: {new_content[:50]}..."

    def get_all_knowledge(self):
        """获取知识库所有内容"""
        if self.vectorstore is None:
            return []
        try:
            collection = self.vectorstore._collection
            all_data = collection.get(include=["documents"])
            return all_data.get("documents", [])
        except:
            return []


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