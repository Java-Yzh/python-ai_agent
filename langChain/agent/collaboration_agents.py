import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ================================================================
# 场景：研究员 Agent + 写手 Agent 协作写报告
# ================================================================

# 创建研究员agent
researcher_llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)
researcher_tools = [TavilySearch(max_results=5)]
researcher_agent = create_agent(researcher_llm, researcher_tools)

# 创建写手agent
writer_llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

writer_prompt = ChatPromptTemplate.from_template(
    """你是一个专业写手，根据研究材料撰写报告。

研究材料：
{research_material}

要求：写一篇结构清晰的报告，包含摘要、背景、分析、结论四部分。"""
)

writer_chain = (
    writer_prompt
    | writer_llm
    | StrOutputParser()
)

# 研究员agent与写手agent协作
def collaborative_report(topic: str) -> str:
    """研究员搜集资料 → 写手撰写报告"""
    # 研究员搜索素材
    research_response = researcher_agent.invoke({"messages": [{"role": "user",
                                           "content": f"请收集关于「{topic}」的最新信息，包括定义、现状、重要进展、未来趋势。输出结构化的研究摘要。"}]})
    # 提取最终回答，作为写手agent的输入
    print(research_response)
    # 返回值是一个 dict，核心只有一个 key："messages"，它是一个消息列表，按执行顺序记录了整个过程：
    # {
    #     "messages": [
    #         # [0] 你的输入 —— HumanMessage
    #         HumanMessage(content="请收集关于「机器学习」的最新信息..."),
    #
    #         # [1] LLM 决定调用工具 —— AIMessage（可能有多轮）
    #         AIMessage(
    #             content="",                          # 此时通常为空或简短思考
    #             tool_calls=[                         # LLM 决定调用哪些工具
    #                 {
    #                     "name": "tavily_search",
    #                     "args": {"query": "机器学习最新进展"},
    #                     "id": "call_xxx"             # 调用 ID
    #                 }
    #             ]
    #         ),
    #
    #         # [2] 工具执行结果 —— ToolMessage
    #         ToolMessage(
    #             content="[{'title': '...', 'url': '...', 'content': '...'}]",
    #             name="tavily_search",                # 哪个工具返回的
    #             tool_call_id="call_xxx"              # 对应上面的调用 ID
    #         ),
    #
    #         # ... 可能还有更多 AIMessage ↔ ToolMessage 的循环 ...
    #
    #         # [-1] 最终回答 —— AIMessage（你需要的）
    #         AIMessage(content="关于机器学习的最新信息：\n1. 定义...\n2. 现状...")
    #     ]
    # }
    print("----------------------------------------------------------------------")
    research_material = research_response["messages"][-1].content

    # 写手撰写报告
    report = writer_chain.invoke({"research_material": research_material})
    print(report)

if __name__ == "__main__":
    collaborative_report("机器学习")