# Agent 与人机交互：打断与确认
# 使用 LangGraph 的 interrupt_before 机制实现人工审批
# 流程：用户输入 → Agent 推理 → 暂停等待人工确认 → 继续执行或终止
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

# 创建大模型
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    action: str

# Agent 推理节点：让 LLM 决定要执行什么操作
def agent_node(state: AgentState):
    """Agent 推理：分析用户意图，决定执行什么操作"""
    response = llm.invoke(state["messages"])
    content = response.content
    print(f"\n[Agent 思考] {content[:200]}")

    # 简单判断：根据 LLM 回复中是否包含敏感关键词来决定是否需要审批
    sensitive_keywords = ["删除", "支付", "下单", "永久", "转账"]
    action = "sensitive" if any(kw in content for kw in sensitive_keywords) else "safe"

    return {"messages": [response], "action": action}


# 敏感操作执行节点（需要人工审批才能到达这里）
def execute_sensitive(state: AgentState):
    """执行敏感操作（经过人工确认后）"""
    print("\n[执行] 人工已确认，正在执行敏感操作...")
    return {"messages": []}


# 路由判断：根据 action 决定下一步走向
def should_continue(state: AgentState):
    """路由：safe → 直接结束，sensitive → 进入等待审批节点"""
    if state.get("action") == "sensitive":
        return "wait_approval"
    return END


# 构建图
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("wait_approval", execute_sensitive)
builder.set_entry_point("agent")

builder.add_conditional_edges("agent", should_continue, {
    "wait_approval": "wait_approval",
    END: END,
})
builder.add_edge("wait_approval", END)

# 关键：interrupt_before 让图在到达 wait_approval 节点之前暂停
checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["wait_approval"],  # 进入此节点前暂停，等待人工确认
)


# ── 运行示例 ──────────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}

    # 场景1：普通请求（不会触发中断）
    print("=" * 50)
    print("场景1：普通请求")
    print("=" * 50)
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，今天天气怎么样？")], "action": ""},
        config=config,
    )
    print(f"\n图是否还有下一步？{graph.get_state(config).next}")

    # 场景2：敏感请求（会触发中断）
    print("\n" + "=" * 50)
    print("场景2：敏感请求 —— 会触发中断")
    print("=" * 50)
    config2 = {"configurable": {"thread_id": "demo-2"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="请帮我删除所有过期数据，永久清除")], "action": ""},
        config=config2,
    )

    # 检查图是否暂停了
    current_state = graph.get_state(config2)
    print(f"\n图暂停了！等待人工确认。下一步节点: {current_state.next}")

    if current_state.next:
        # 模拟人工确认：approve → 继续执行
        user_input = input("\n是否批准此操作？(y/n): ").strip().lower()
        if user_input == "y":
            print("\n--- 人工批准，继续执行 ---")
            graph.invoke(None, config=config2)  # 传入 None 表示"批准，继续"
            print("操作已完成！")
        else:
            print("\n--- 人工拒绝，终止操作 ---")
            # 通过更新状态来跳过敏感节点，直接结束
            graph.update_state(config2, {"action": "cancelled"})
            print("操作已取消！")
