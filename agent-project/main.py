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
    try:
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase(persist_dir=get_resource_path("chroma_db"))

        # 添加测试数据（检查 vectorstore 是否存在）
        if kb.vectorstore is None or kb.vectorstore._collection.count() == 0:
            print("添加测试数据...")
            kb.add_documents([
                # 基础信息
                "我的名字叫张三，今年25岁，是一名软件工程师，住在上海。",
                "我擅长Python编程，熟悉Django、FastAPI等框架。",
                "我目前在一家互联网公司工作，主要做后端开发。",
                # 爱好 - 用多个表述增加匹配机会
                "我的爱好是爬山和看电影，周末喜欢去爬山，也喜欢在家看科幻电影。",
                "我周末喜欢爬山锻炼身体，也喜欢看电影特别是科幻大片。",
            ])

        def search_knowledgebase(query: str) -> str:
            results = kb.search(query, k=3)
            if not results:
                return "知识库中没有相关信息"
            return "\n\n".join(results)

        def learn_knowledge(text: str) -> str:
            """让 Agent 学习新知识，添加到知识库中"""
            if not text or not text.strip():
                return "学习内容不能为空"
            result = kb.learn(text)
            return result

        kb_tools = [
            Tool.from_function(
                func=search_knowledgebase,
                name="knowledge_base_search",
                description="当用户问关于个人、公司、简历、私人信息等问题时使用此工具。"
            ),
            Tool.from_function(
                func=learn_knowledge,
                name="knowledge_base_learn",
                description="当用户明确说'记住'、'学习'、'更新信息'、'以后要记住'等内容时，或者用户提供了新的个人信息（如名字、爱好、工作等）时使用此工具。参数 text 是要学习的具体内容。"
            )
        ]
        return kb_tools
    except Exception as e:
        print(f"知识库加载失败: {e}")
        return None

# 延迟初始化 kb_tools
kb_tools = None

# 2. 工具列表
tool_list = [search_tool]

# 3. 运行 - 手动选择工具
def run_agent(query: str):
    global kb_tools

    # 延迟初始化知识库
    if kb_tools is None:
        print("加载知识库...")
        kb_tools = get_kb_tool()

    # 获取知识库搜索工具
    kb_search_tool = kb_tools[0] if kb_tools else None
    kb_learn_tool = kb_tools[1] if kb_tools else None

    # 根据关键词判断问题类型
    personal_keywords = ["我", "你", "名字", "爱好", "喜欢", "工作", "住哪", "擅长", "周末"]
    search_keywords = ["天气", "新闻", "搜索", "现在"]
    learn_keywords = ["记住", "学习", "更新", "以后要", "请记"]

    is_personal = any(kw in query for kw in personal_keywords)
    is_search = any(kw in query for kw in search_keywords)
    is_learn = any(kw in query for kw in learn_keywords)

    if is_learn and kb_learn_tool:
        # 学习新知识
        learn_result = kb_learn_tool.func(query)
        return f"[已学习] {learn_result}"

    if is_personal and kb_search_tool:
        # 知识库搜索
        kb_result = kb_search_tool.func(query)
        # 让 LLM 根据知识库内容回答
        prompt = f"""从以下知识库内容中找出最相关的信息回答用户问题，直接给出答案：

知识库：
{kb_result}

问题：{query}

回答："""
    elif is_search:
        # 联网搜索
        search_result = search_tool.invoke(query)
        prompt = f"根据搜索结果回答用户问题（用中文）：\n\n{search_result}\n\n问题：{query}"
    else:
        # 默认知识库优先
        if kb_search_tool:
            kb_result = kb_search_tool.func(query)
            if kb_result and "没有相关信息" not in kb_result:
                prompt = f"根据知识库回答：{kb_result}"
            else:
                search_result = search_tool.invoke(query)
                prompt = f"根据搜索结果回答：{search_result}"
        else:
            search_result = search_tool.invoke(query)
            prompt = f"回答：{search_result}"

    # 用 LLM 生成回答
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import scrolledtext, messagebox

    # 创建GUI窗口
    root = tk.Tk()
    root.title("AI Agent - 联网搜索 + 知识库问答")
    root.geometry("700x600")

    # 标题
    title_label = tk.Label(root, text="AI Agent 助手", font=("微软雅黑", 16, "bold"))
    title_label.pack(pady=10)

    # 说明
    info_label = tk.Label(root, text="支持联网搜索 + 本地知识库问答", font=("微软雅黑", 10))
    info_label.pack(pady=5)

    # 输入区域
    input_frame = tk.Frame(root)
    input_frame.pack(pady=10, padx=20, fill=tk.X)

    tk.Label(input_frame, text="请输入问题:").pack(anchor=tk.W)

    input_box = tk.Entry(input_frame, font=("微软雅黑", 12))
    input_box.pack(fill=tk.X, pady=5)

    # 发送按钮
    def send_query():
        query = input_box.get().strip()
        if not query:
            messagebox.showwarning("提示", "请输入问题")
            return

        send_btn.config(state=tk.DISABLED)
        output_text.insert(tk.END, f"\n你: {query}\n")
        output_text.see(tk.END)
        input_box.delete(0, tk.END)

        root.update()
        try:
            response = run_agent(query)
            output_text.insert(tk.END, f"\nAgent: {response}\n")
            output_text.see(tk.END)
        except Exception as e:
            output_text.insert(tk.END, f"\n错误: {str(e)}\n")
        finally:
            send_btn.config(state=tk.NORMAL)

    send_btn = tk.Button(input_frame, text="发送", command=send_query, font=("微软雅黑", 11), bg="#4CAF50", fg="white")
    send_btn.pack(pady=5)

    # 清空按钮
    def clear_output():
        output_text.delete(1.0, tk.END)

    clear_btn = tk.Button(input_frame, text="清空", command=clear_output, font=("微软雅黑", 10))
    clear_btn.pack(pady=2)

    # 输出区域
    output_frame = tk.Frame(root)
    output_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    tk.Label(output_frame, text="回答:").pack(anchor=tk.W)

    output_text = scrolledtext.ScrolledText(output_frame, font=("微软雅黑", 11), height=15)
    output_text.pack(fill=tk.BOTH, expand=True)

    # 回车发送
    def on_enter(event):
        send_query()

    input_box.bind("<Return>", on_enter)

    # 初始化提示
    output_text.insert(tk.END, "=== Agent 已启动 ===\n")
    output_text.insert(tk.END, "支持：联网搜索 + 本地知识库问答\n")
    output_text.insert(tk.END, "请在上方输入问题并点击发送\n")

    root.mainloop()