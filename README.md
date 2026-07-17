# Python 与 AI Agent 开发学习

本项目是 Python 编程与 AI Agent 开发的学习仓库，从基础语法到 LLM API 调用、LangChain 框架、RAG 检索增强与 Agent 智能体，逐步积累开发经验。

## 项目结构

```
.
├── Introduction/               # Python 基础语法练习
│   ├── test1.py               # 输入输出、算术运算（BMI 计算器）
│   ├── test2.py               # 条件判断（if/elif/else）
│   ├── test3.py               # 嵌套循环、print 格式化（sep/end）
│   ├── test4.py               # while 循环
│   ├── test5.py               # 循环累加（for + while）
│   ├── test6.py               # 列表操作（索引、切片、增删）
│   ├── test7.py               # 字典操作（键值对、遍历）
│   ├── test8.py               # 元组操作（不可变性、拆包）
│   ├── test9.py               # 函数定义与默认参数
│   ├── test10.py              # 文件读写（w/a/r 模式）
│   ├── test11.py              # 综合实战：CSV 个人收支管家
│   ├── 档案.txt               # test10 生成的示例文件
│   └── 账本.csv               # test11 生成的示例文件
├── advanced/                  # Python 进阶与 OpenAI SDK
│   ├── config_env.py          # python-dotenv 环境变量管理
│   ├── openai_sync_demo.py    # OpenAI 同步调用、Function Calling、JSON 模式
│   ├── openai_async_stream.py # 异步并行调用、流式输出、tenacity 重试机制
│   └── .env                   # 环境变量配置（已 gitignore）
├── langChain/                 # LangChain 框架学习
│   ├── first.py               # ChatOpenAI 基础：多轮对话、批量调用、流式输出
│   ├── prompts.py             # Prompt Template 提示词模板
│   ├── output_parser.py       # 输出解析器：StrOutputParser、JsonOutputParser
│   ├── lcel.py                # LCEL 链式调用：管道符、并行、条件分支、数据透传
│   ├── RAG/                   # RAG 检索增强生成
│   │   ├── vector_storage.py              # 向量存储：文档加载→分割→嵌入→Chroma
│   │   ├── rag_basic.py                   # 基础相似度检索
│   │   ├── rag_mmr.py                     # MMR 最大边际相关性（多样性）
│   │   ├── rag_multiQueryRetriever.py     # 多查询检索（LLM 改写问题）
│   │   ├── rag_contextualCompressionRetriever.py # 上下文压缩（LLM 提取相关片段）
│   │   ├── rag_embeddingsFilter.py        # 嵌入相似度过滤（轻量替代 LLM 压缩）
│   │   ├── rag_ensembleRetriever.py       # 混合检索（向量 + BM25 关键词）
│   │   ├── rag_longContextReorder.py      # 长上下文重排序（解决"中间遗忘"）
│   │   ├── rag_multiVectorRetriever.py   # 多向量检索（摘要索引、原文返回）
│   │   ├── rag_parentDocumentRetriever.py # 父子文档检索（子块检索、父块返回）
│   │   ├── rag_selfQueryRetriever.py     # 自查询检索（自然语言转结构化过滤）
│   │   ├── doc.md                       # RAG 示例文档
│   │   └── chroma_db/                   # Chroma 向量数据库（已 gitignore）
│   ├── agent/                 # Agent 智能体开发
│   │   ├── first_demo.py                 # create_agent + ReAct 模式入门
│   │   ├── handwriting_reAct_agent.py     # 手写 ReAct 循环（OpenAI SDK 原生）
│   │   ├── tool.py                       # @tool 装饰器、StructuredTool、Schema
│   │   ├── toolkit.py                    # SQLDatabaseToolkit 工具包
│   │   ├── memory.py                     # MemorySaver 会话记忆管理
│   │   ├── interrupt.py                  # 人机交互：打断与人工审批
│   │   ├── exception.py                  # 错误处理：工具级重试 + Agent 级兜底
│   │   ├── log.py                        # 可观测性：回调追踪与运行指标
│   │   └── collaboration_agents.py       # 多 Agent 协作（研究员 + 写手）
│   └── .env                   # 环境变量配置（已 gitignore）
├── langGraph/                 # LangGraph 学习（待补充）
├── streamlit/                 # Streamlit Web 应用学习（待补充）
├── main.py
├── pyproject.toml             # 项目依赖管理（uv）
└── .python-version            # Python 版本锁定（>=3.12）
```

## 环境准备

1. 安装 [uv](https://docs.astral.sh/uv/) 包管理器
2. 克隆项目后安装依赖：
   ```bash
   uv sync
   ```
3. 配置环境变量文件：

   **`advanced/.env`**（OpenAI SDK 示例）：
   ```env
   OPENAI_API_KEY=your-api-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4o-mini
   # 以下为异步流式示例所需的多模型配置
   OPENAI_MODEL_FLASH=your-flash-model
   OPENAI_MODEL_PRO=your-pro-model
   MAX_RETRIES=3
   ```

   **`langChain/.env`**（LangChain 示例，默认使用 DeepSeek）：
   ```env
   DEEPSEEK_API_KEY=your-api-key
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   DEEPSEEK_MODEL_PRO=deepseek-v4-pro
   DEEPSEEK_MODEL_FLASH=deepseek-v4-flash
   # Agent 搜索工具所需
   TAVILY_API_KEY=your-tavily-api-key
   ```

   > 也支持其他兼容 OpenAI 格式的服务，如 DeepSeek、智谱、阿里百炼等。
   > RAG 示例使用 HuggingFace 本地嵌入模型（`sentence-transformers/all-MiniLM-L6-v2`），首次运行需下载模型。

## 运行方式

```bash
# === Python 基础 ===
python Introduction/test1.py    # BMI 计算器
python Introduction/test11.py   # CSV 个人收支管家

# === Python 进阶 ===
python advanced/config_env.py           # 环境变量管理
python advanced/openai_sync_demo.py     # 同步调用 & Function Calling
python advanced/openai_async_stream.py  # 异步 & 流式输出 & 重试

# === LangChain 基础 ===
python langChain/first.py        # 模型调用入门
python langChain/prompts.py      # 提示词模板
python langChain/output_parser.py # 输出解析器
python langChain/lcel.py         # LCEL 链式调用

# === RAG 检索增强 ===
python langChain/RAG/vector_storage.py            # 先运行：构建向量库
python langChain/RAG/rag_basic.py                 # 基础检索
python langChain/RAG/rag_mmr.py                   # MMR 多样性检索
python langChain/RAG/rag_multiQueryRetriever.py   # 多查询检索
python langChain/RAG/rag_selfQueryRetriever.py    # 自查询检索
# ... 其他 RAG 示例同理

# === Agent 智能体 ===
python langChain/agent/first_demo.py              # Agent 入门
python langChain/agent/handwriting_reAct_agent.py # 手写 ReAct
python langChain/agent/tool.py                    # 工具定义
python langChain/agent/memory.py                  # 会话记忆
python langChain/agent/interrupt.py               # 人工审批
python langChain/agent/collaboration_agents.py    # 多 Agent 协作
```

## 学习内容

### Python 基础（Introduction/）

- 数据类型与运算：输入输出、算术运算、变量
- 流程控制：条件判断（if/elif/else）、循环（for/while）
- 核心数据结构：列表、字典、元组的特性与操作
- 函数：定义、默认参数、返回值
- 文件 I/O：读写模式（w/a/r）、CSV 处理
- 综合实战：基于 CSV 的个人收支管理应用

### Python 进阶（advanced/）

- 环境管理：`python-dotenv` 管理敏感配置、`uv` 管理项目依赖
- OpenAI SDK 同步调用：Chat Completions API、消息角色（system/user/assistant/tool）
- Function Calling：工具注册、Schema 定义、多轮工具调用流程
- 结构化输出：`response_format` JSON 模式
- 异步编程：`asyncio.gather` 并行调用多模型
- 流式输出：SSE 逐块处理
- 生产容错：`tenacity` 指数退避重试、错误分类

### LangChain 框架（langChain/）

- **模型调用**：`ChatOpenAI` 初始化、invoke/batch/stream 三种调用方式
- **提示词工程**：`ChatPromptTemplate`、`MessagesPlaceholder` 多轮对话模板
- **输出解析**：`StrOutputParser`、`JsonOutputParser` + Pydantic 结构化校验
- **LCEL 链式调用**：管道符 `|` 串联组件、`RunnableParallel` 并行、`RunnableBranch` 条件分支、`RunnablePassthrough` 数据透传

### RAG 检索增强（langChain/RAG/）

- **向量存储**：文档加载（PDF/Text/CSV/Markdown/Web）→ 分割（`RecursiveCharacterTextSplitter`）→ 嵌入（HuggingFace）→ Chroma 向量库
- **基础检索**：相似度检索
- **高级检索策略**：
  - MMR：兼顾相关性与多样性
  - MultiQueryRetriever：LLM 自动改写多角度查询
  - ContextualCompressionRetriever：LLM 提取相关片段
  - EmbeddingsFilter：嵌入相似度过滤（轻量替代）
  - EnsembleRetriever：向量 + BM25 混合检索
  - LongContextReorder：重排序解决"中间遗忘"
  - MultiVectorRetriever：摘要索引 + 原文返回
  - ParentDocumentRetriever：子块检索 + 父块返回
  - SelfQueryRetriever：自然语言转 metadata 过滤

### Agent 智能体（langChain/agent/）

- **Agent 入门**：`create_agent` + ReAct（Reasoning + Acting）模式
- **手写 ReAct**：用 OpenAI SDK 原生实现思考-行动-观察循环
- **工具开发**：`@tool` 装饰器、`StructuredTool.from_function`、Pydantic Schema
- **工具包**：`SQLDatabaseToolkit` 自动生成数据库查询工具
- **会话记忆**：`MemorySaver` + `thread_id` 多会话隔离
- **人机交互**：LangGraph `interrupt_before` 人工审批流程
- **错误处理**：工具级重试（指数退避）+ Agent 级兜底降级
- **可观测性**：`BaseCallbackHandler` 自定义回调、运行指标采集
- **多 Agent 协作**：研究员 Agent + 写手 Agent 分工协作

## 技术栈

| 分类 | 技术 |
|------|------|
| 语言 & 版本 | Python >= 3.12 |
| 包管理 | uv |
| LLM SDK | openai |
| 框架 | LangChain（langchain-core / langchain-openai / langchain-community / langchain-huggingface / langchain-tavily / langchain-chroma） |
| 向量数据库 | ChromaDB |
| 嵌入模型 | sentence-transformers（all-MiniLM-L6-v2，本地运行） |
| 搜索工具 | Tavily Search |
| 文档加载 | unstructured、pypdf、beautifulsoup4 |
| 容错 | tenacity |
| 配置管理 | python-dotenv |
