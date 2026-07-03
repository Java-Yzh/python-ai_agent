# Python 与 AI Agent 开发学习

本项目是 Python 编程与 AI Agent 开发的学习仓库，从基础语法到 LLM API 调用，逐步积累开发经验。

## 项目结构

```
.
├── Introduction/       # Python 基础语法练习（数据类型、流程控制、函数、文件操作等）
├── advanced/           # 进阶内容（环境变量配置、OpenAI SDK 调用、Function Calling 等）
├── main.py
├── pyproject.toml      # 项目依赖管理（uv）
└── .python-version     # Python 版本锁定（>=3.12）
```

## 环境准备

1. 安装 [uv](https://docs.astral.sh/uv/) 包管理器
2. 克隆项目后安装依赖：
   ```bash
   uv sync
   ```
3. 在 `advanced/` 目录下创建 `.env` 文件，配置以下内容：
   ```env
   OPENAI_API_KEY=your-api-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4o-mini
   ```
   > 也支持其他兼容 OpenAI 格式的服务，如 DeepSeek、智谱、阿里百炼等。

## 运行方式

```bash
# 基础练习
python Introduction/test1.py

# 环境变量配置示例
python advanced/config_env.py

# OpenAI 同步调用 & Function Calling 示例
python advanced/openai_sync_demo.py
```

## 学习内容

- **Python 基础**：数据类型、条件判断、循环、函数、文件 I/O、字典与列表等
- **环境管理**：`python-dotenv` 管理敏感配置、`uv` 管理项目依赖
- **AI Agent 开发**：OpenAI Chat Completions API、Function Calling（工具调用）、多轮对话上下文管理
