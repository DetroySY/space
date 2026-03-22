# 从零开始：搭建一个本地 AI 助手（Python + LangGraph + 向量数据库）

> 一个适合 Python 入门者学习的大语言模型应用实战

## 前言

你是否想过自己做一个 AI 助手？不需要很复杂，就像 Siri 小爱同学那样，能联网查资料、能回答个人问题就够了。

本文就带你用 **Python** 实现这样一个应用，整个过程只需要几百行代码。

---

## 最终效果

我们的应用可以：
- 🌐 **联网搜索** - 实时查询天气、新闻等信息
- 📚 **本地知识库** - 回答关于你个人的问题（比如"我住在哪里"）
- 💻 **打包成 exe** - 变成indows 桌面程序发给朋友用

---

## 技术栈

| 技术 | 作用 |
|------|------|
| Python 3.10+ | 编程语言 |
| LangGraph | Agent 框架，让 AI 自己选择用什么工具 |
| Chroma | 向量数据库，存储本地知识 |
| SiliconFlow | 国内可用的 API 服务商（兼容 OpenAI） |
| PyInstaller | 打包成 exe |

---

## 1. 环境准备

首先安装依赖：

```bash
pip install langchain langchain-community langchain-siliconflow chromadb python-dotenv tavily
```

再创建一个 `.env` 文件，填入你的 API Key（ SilberFlow 免费注册送额度）：

```
SILICONFLOW_API_KEY=你的API密钥
```

> 提示：SiliconFlow 是国内镜像服务，访问速度快，支持 Qwen、DeepSeek 等多种模型。

---

## 2. 知识库：让 AI 记得你的事

我们用 **Chroma** 向量数据库来存储个人知识。原理很简单：把文字转成向量，相似的内容会"靠得近"，搜索时找最相似的就行。

```python
# knowledge_base.py
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 用这个轻量中文 Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

class KnowledgeBase:
    def __init__(self, persist_dir="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts):
        """添加文档"""
        self.vectorstore.add_texts(texts=texts)

    def search(self, query, k=3):
        """搜索最相似的 k 条"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
```

**测试一下：**

```python
kb = KnowledgeBase()
kb.add_documents([
    "我的名字叫张三，住在上海。",
    "我在一家互联网公司做后端开发。"
])

print(kb.search("我是谁"))
# 输出: ['我的名字叫张三，住在上海。', ...]
```

---

## 3. Agent：让 AI 自己选工具

现在有搜索工具（联网）和知识库工具（本地），我们需要一个"大脑"来决定什么时候用什么。

**LangGraph** 的 ReAct Agent 就是干这个的：

```python
# main.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults

# 1. 初始化大模型
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)

# 2. 搜索工具
search_tool = TavilySearchResults(max_results=3)

# 3. 知识库工具
def get_kb_tool():
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase(persist_dir="chroma_db")

    def search_knowledgebase(query: str) -> str:
        results = kb.search(query, k=3)
        if not results:
            return "知识库中没有相关信息"
        return "\n\n".join(results)

    return Tool.from_function(
        func=search_knowledgebase,
        name="knowledge_base_search",
        description="当用户问个人、公司、简历等问题时使用"
    )

# 4. 创建 Agent
tools = [search_tool, get_kb_tool()]
agent = create_react_agent(llm, tools)

# 5. 运行
system_msg = """你是一个智能助手：
- 问实时信息（天气、新闻）用 tavily_search
- 问私人信息用 knowledge_base_search"""

result = agent.invoke({
    "messages": [
        ("system", system_msg),
        ("user", "今天北京天气怎么样？")
    ]
})

print(result["messages"][-1].content)
```

---

## 4. 打包成桌面应用

用 **PyInstaller**，一行命令搞定：

```bash
pyinstaller --onefile --name AgentApp main.py
```

生成的 `dist/AgentApp.exe` 就可以发给朋友用了！

---

## 完整代码结构

```
agent-project/
├── main.py              # 主程序
├── knowledge_base.py    # 知识库
├── .env                 # API 密钥
├── chroma_db/           # 向量数据库（自动生成）
└── requirements.txt     # 依赖
```

---

## 运行效果

```
=== Agent 已启动 ===
支持：联网搜索 + 本地知识库问答
输入 q 退出

你: 我是谁？
Agent: 根据知识库信息，您的名字叫张三，是一名软件工程师，住在上海。

你: 今天上海天气怎么样？
Agent: 今天上海天气晴朗，气温 15-22°C...
```

---

## 进阶方向

学会了基础，你可以尝试：
- 🖼️ **加个 UI** - 用 Gradio 或 PyQt 做个界面
- 💾 **知识库持久化** - 把 TXT/MD 文件批量导入
- 🔧 **更多工具** - 接入计算器、日历等
- 🤖 **换更强模型** - 试试 DeepSeek 或 GPT-4

---

## 总结

这就是做一个 AI 助手的全部核心：
1. **Embedding** - 把文字转成向量
2. **向量数据库** - 存和搜向量
3. **Agent** - 让 AI 自主选择工具
4. **API** - 调用大模型能力

全部加在一起，就是几百行代码。感兴趣的话，快去试试吧！