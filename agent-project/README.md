# AI Agent - 联网搜索 + 本地知识库问答

一个基于 LangGraph 的智能助手，支持联网搜索和本地知识库问答，并具有动态学习能力。

## 功能特性

- **联网搜索**：集成 Tavily 搜索，实时获取新闻、天气等信息
- **本地知识库**：基于 Chroma + Embedding 的向量知识库
- **动态学习**：支持边聊边学，自动判断新增或更新知识
- **GUI 界面**：简洁的图形界面，操作方便

## 技术栈

- LangGraph / LangChain
- ChatOpenAI (SiliconFlow API)
- Chroma (向量数据库)
- Qwen3-Embedding-0.6B (嵌入模型)
- Tavily Search (搜索)

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在 `agent-project` 目录下创建 `.env` 文件：

```env
SILICONFLOW_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## 运行

```bash
python main.py
```

## 使用方式

### 联网搜索
输入新闻、天气等关键词，自动调用 Tavily 搜索。

### 知识库问答
问关于个人、公司、简历等私人信息时，自动从知识库检索答案。

### 动态学习
告诉 Agent "记住" 或 "学习" 新信息，它会自动添加到知识库：

```
记住我叫李四，今年30岁
更新一下，我今年31岁了
```

## 项目结构

```
agent-project/
├── main.py              # 主程序 (GUI)
├── knowledge_base.py    # 知识库类
├── requirements.txt     # 依赖
├── .env                 # API 密钥配置
└── chroma_db/           # 向量数据库存储目录
```

## License

MIT
