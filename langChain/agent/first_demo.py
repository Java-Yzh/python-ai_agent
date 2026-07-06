# Agent工作原理
# 用户输入 → Agent（LLM推理）→ 决定调用哪个 Tool → 执行 Tool → 观察结果
#          ↑                                                    ↓
#          └──────────────── 循环直到完成 ←───────────────────────┘

# 核心三要素：
# LLM，推理引擎，决定"下一步做什么"
# Tools，具体执行能力（搜索、计算、数据库查询等）
# Orchestrator，编排循环，管理 Agent 的思考-行动-观察循环

# langchain-core → 抽象接口（BaseLLM、BaseEmbedding等），不依赖任何第三方
# langchain → 编排逻辑（链、代理、记忆），只依赖 core
# langchain-{provider} → 官方维护的单个服务商集成（如 langchain-openai、langchain-anthropic）
# langchain_community → 剩下的"杂项"集成，社区维护，门槛低、更新快

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# Tavily 是一个专为 AI Agent 和 LLM 优化的搜索 API 服务，定位是"AI 的搜索引擎"——它不是给人用的，而是给大模型和智能体当"上网工具"的。
from langchain_tavily import TavilySearch

# 从当前工作目录开始（即你运行 Python 脚本时所在的目录），然后逐级向上查找，直到找到第一个 .env 文件为止
load_dotenv()

# 初始化LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 定义工具（tools）
# 搜索工具
# TavilySearch 初始化时，自动从环境变量中读取 TAVILY_API_KEY
search = TavilySearch(max_results=3)

# 计算工具
@tool
def calculator(expression: str) -> str:
    """执行数学计算。输入：数学表达式字符串，例如 '2 + 3 * 4'"""
    try:
        # 安全计算（仅允许数学表达式）
        import ast
        import operator

        allowed_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }

        def safe_eval(node):
            if isinstance(node, ast.Expression):
                return safe_eval(node.body)
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](
                    safe_eval(node.left), safe_eval(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](safe_eval(node.operand))
            elif isinstance(node, ast.Constant):
                return node.value
            raise ValueError(f"不支持的操作: {type(node)}")

        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

tools = [search, calculator]

# 创建agent
# ReAct 模式：思考→行动→观察→循环，create_agent 是 create_react_agent 的替代者，
# 底层统一跑在 LangGraph 上，自带 ReAct（Reasoning + Acting）循环
# LLM 自己决定什么时候该调工具、调哪个工具、什么时候停止
# 整个 ReAct 循环的核心就是 LLM 自己决定调不调工具，框架根据 tool_calls 是否为空来决定继续还是结束，完全自动。
# 不需要手动管理上下文以及工具调用，整个流程不需要手动控制
agent = create_agent(llm, tools)

# 运行agent
response = agent.invoke({
    "messages": [HumanMessage(content="2024年诺贝尔物理学奖得主是谁？把他的年龄除以 2 是多少？")]
})

# 获取最终回答
# 这段代码是遍历 Agent 返回的所有消息，逐条打印有内容的消息。具体拆解：
# response["messages"] — Agent 运行过程中产生的所有消息列表，包括：
# HumanMessage（用户输入）
# AIMessage（LLM 的思考/回答，可能包含 tool_calls）
# ToolMessage（工具执行结果）
# hasattr(msg, "content") and msg.content — 过滤掉没有 content 或 content 为空的消息（比如某些 AIMessage 只包含 tool_calls，content 为空）
# msg.__class__.__name__ — 获取消息类型名称，如 AIMessage、ToolMessage 等
# msg.content[:200] — 截取内容前 200 个字符，防止输出过长
for msg in response["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(f"[{msg.__class__.__name__}] {msg.content[:200]}")
# print(response)