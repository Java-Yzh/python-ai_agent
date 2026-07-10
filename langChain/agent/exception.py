import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

tools = [TavilySearch(max_results=3)]

@tool
# 方案一：工具级错误处理
def robust_search(query: str) -> str:
    """搜索互联网，带重试机制"""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 实际搜索逻辑
            search_tool = TavilySearch(max_results=3)
            return search_tool.invoke({"query": query})
        except Exception as e:
            if attempt == max_retries - 1:
                return f"搜索失败（已重试{max_retries}次）: {str(e)}"
            time.sleep(2 ** attempt)  # 指数退避
    return "未知错误"

# 方案二：Agent 级错误处理
def agent_with_fallback(user_input: str) -> str:
    """Agent 调用带兜底"""
    try:
        agent1 = create_agent(llm, tools)
        result = agent1.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        return result["messages"][-1].content
    except Exception as e:
        # 兜底：降级为直接提问LLM
        result = llm.invoke({"messages": [{"role": "user", "content": user_input}]})
        return result.content

if __name__ == "__main__":
    # 工具级错误处理
    # agent = create_agent(llm, [robust_search])
    # res = agent.invoke(
    #     {"messages": [{"role": "user", "content": "2024年诺贝尔物理学奖得主是谁？"}]}
    # )
    # print(res["messages"][-1].content)

    # Agent级兜底处理
    res = agent_with_fallback("2024年诺贝尔物理学奖得主是谁？")
    print(res)