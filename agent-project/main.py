"""
Agent 项目 - 联网搜索 + 本地知识库问答
"""
import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer, 'strict')

# 获取资源目录（支持 exe 打包和开发环境）
def get_resource_path(relative_path):
    """获取资源文件路径"""
    if getattr(sys, 'frozen', False):
        # 打包成 exe 后，资源在 _MEIPASS 目录下
        base_path = sys._MEIPASS
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# 加载 .env
env_path = get_resource_path('.env')
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

# 1. 初始化
print("初始化中...")

# 大模型
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
)

# 搜索工具
search_tool = TavilySearchResults(max_results=3)

# 知识库工具（动态导入，在函数内初始化）
def get_kb_tool():
    from knowledge_base import KnowledgeBase
    kb = KnowledgeBase(persist_dir=get_resource_path("chroma_db"))

    # 添加测试数据
    if kb.vectorstore._collection.count() == 0:
        print("添加测试数据...")
        kb.add_documents([
            "我的名字叫张三，今年25岁，是一名软件工程师，住在上海。",
            "我擅长Python编程，熟悉Django、FastAPI等框架。",
            "我周末喜欢爬山和看电影，尤其是科幻电影。",
            "我目前在一家互联网公司工作，主要做后端开发。",
        ])

    def search_knowledgebase(query: str) -> str:
        results = kb.search(query, k=3)
        if not results:
            return "知识库中没有相关信息"
        return "\n\n".join(results)

    return Tool.from_function(
        func=search_knowledgebase,
        name="knowledge_base_search",
        description="当用户问关于个人、公司、简历、私人信息等问题时使用此工具。"
    )

# 延迟初始化 kb_tool
kb_tool = None

# 2. 创建 Agent
tools = [search_tool]
agent = create_react_agent(llm, tools)

# 3. 运行
def run_agent(query: str):
    global kb_tool, agent

    # 延迟初始化知识库
    global kb_tool
    if kb_tool is None:
        print("加载知识库...")
        kb_tool = get_kb_tool()
        tools.append(kb_tool)
        # 重新创建 agent
        agent = create_react_agent(llm, tools)

    system_msg = """你是一个智能助手，需要根据用户问题选择合适的工具：

- 当用户问实时信息、新闻、天气、搜索相关内容时，使用 tavily_search 工具
- 当用户问个人、公司、简历等私人信息时，使用 knowledge_base_search 工具

请根据问题内容自主选择使用哪个工具。"""

    result = agent.invoke({
        "messages": [
            ("system", system_msg),
            ("user", query)
        ]
    })
    return result["messages"][-1].content

if __name__ == "__main__":
    print("\n=== Agent 已启动 ===")
    print("支持：联网搜索 + 本地知识库问答")
    print("输入 q 退出\n")

    while True:
        query = input("你: ")
        if query.lower() == "q":
            break

        response = run_agent(query)
        print(f"\nAgent: {response}\n")