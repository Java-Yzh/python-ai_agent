# Agent 记忆管理
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

@tool
def dummy_tool(query: str) -> str:
    """示例工具，请替换为实际工具"""
    return f"结果: {query}"

tools = [dummy_tool]

# 创建带记忆的 Agent
memory = MemorySaver()
agent_with_memory = create_agent(
    llm,
    tools,
    checkpointer=memory  # 关键：注入 checkpointer
)

# 使用 thread_id 区分不同会话
config = {"configurable": {"thread_id": "user-session-001"}}

# 第一轮
resp1 = agent_with_memory.invoke(
    {"messages": [("user", "我叫张三，我喜欢 Python")]},
    config=config
)
print(resp1["messages"][-1].content)

print("----------------------------------------------------------------------")

# 第二轮
resp2 = agent_with_memory.invoke(
    {"messages": [("user", "我叫什么名字？我喜欢什么编程语言")]},
    config=config
)
print(resp2["messages"][-1].content)

print("----------------------------------------------------------------------")

# 不同 thread_id = 独立会话
config2 = {"configurable": {"thread_id": "user-session-002"}}
resp3 = agent_with_memory.invoke(
    {"messages": [("user", "我叫什么名字？")]},
    config=config2  # 新会话，没有历史记忆
)
print(resp3["messages"][-1].content)