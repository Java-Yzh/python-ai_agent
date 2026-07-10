# Agent 可观测性：追踪与日志
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.callbacks import BaseCallbackHandler
import time

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

tools = [TavilySearch(max_results=3)]

# 方案一：LangSmith 追踪（推荐生产环境）
# 设置环境变量后自动追踪：
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=ls__your-key
# export LANGCHAIN_PROJECT=my-agent-project


# 方案二：自定义回调
# 自定义回调处理器 —— 基于 LangChain 的观察者模式实现 Agent 可观测性。
#
# 继承 BaseCallbackHandler 基类，重写其中的钩子方法（hook methods）。
# LangChain 框架会在 Agent 运行的不同生命周期阶段自动调用这些方法，
# 无需手动触发，从而在不侵入 Agent 核心逻辑的前提下收集运行指标。
#
# 初始化时定义三个状态变量作为累加器，在整个 Agent 运行周期内持续记录：
#   - tool_calls: 记录工具被调用的总次数
#   - total_tokens: 记录 token 消耗总量（当前未填充，可扩展）
#   - start_time: 记录 Chain 开始执行的时间戳
#
# 各钩子方法的触发时机与作用：
#   - on_chain_start: Chain 开始执行时触发，记录当前时间作为计时起点。
#   - on_tool_start: 每次 Agent 调用工具之前触发，将 tool_calls 计数加 1。
#   - on_tool_end: 每次工具调用结束后触发，当前为空实现，预留扩展。
#   - on_chain_end: Chain 执行完毕时触发，用当前时间减去开始时间计算总耗时，
#     并打印耗时和工具调用次数的汇总信息。
#
# 方法签名中使用 *args, **kwargs 的原因：
#   基类的钩子方法会传入多个参数（如 serialized、inputs、run_id 等），
#   但这里并不需要用到这些参数，使用通配参数接收可以避免签名不匹配报错，
#   同时保持与基类接口的兼容性。
#
# 使用方式：
#   将回调实例传入 Agent 的 config 中，框架会自动在对应时机触发钩子：
#     callback = AgentMetricsCallback()
#     agent.invoke(input, config={"callbacks": [callback]})
#   运行结束后会输出类似：[Agent 指标] 耗时: 3.45s | 工具调用: 2次

class AgentMetricsCallback(BaseCallbackHandler):
    """自定义回调，记录 Agent 运行指标"""
    def __init__(self):
        self.tool_calls = 0
        self.total_tokens = 0
        self.start_time = None
        self._root_run_id = None

    # on_chain_start 方法参数说明：
    #   - self: 回调实例本身，用于访问 self.tool_calls、self.start_time 等实例变量。
    #   - serialized: 被触发的 Runnable 的序列化元信息（dict），包含 id、name、type 等字段。
    #     传统 Chain 会在这里携带 name，但 LangGraph 节点不携带，所以需从 kwargs 取 name。
    #   - *args: 位置参数收集器，把基类传入的额外位置参数（如 inputs）打包成元组，
    #     不接收也不会报错，起到向前兼容的作用。
    #   - **kwargs: 关键字参数收集器，把基类传入的额外关键字参数打包成字典，
    #     包含 run_id、parent_run_id、name、tags 等信息，按需通过 kwargs.get() 取用。

    # * 和 ** 是 Python 的参数收集语法。
    # * 把多余的位置参数打包成一个元组（tuple）
    # ** 把多余的关键字参数打包成一个字典（dict）
    # 位置参数 — 按「位置顺序」对应：greet("张三", 18)
    # 关键字参数 — 按「名字」对应：greet(name="张三", age=18)
    def on_chain_start(self, serialized, *args, **kwargs):
        name = kwargs.get("name", "unknown")
        run_id = kwargs.get("run_id")
        parent_id = kwargs.get("parent_run_id")
        tags = kwargs.get("tags", [])
        if parent_id is None:
            self._root_run_id = run_id
            self.start_time = time.time()
        print(f"  [Chain开始] name={name} | tags={tags} | parent={'None' if parent_id is None else '有'}")

    def on_tool_start(self, *args, **kwargs):
        self.tool_calls += 1

    def on_tool_end(self, *args, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id == self._root_run_id:
            elapsed = time.time() - self.start_time
            print(f"[Agent 指标] 耗时: {elapsed:.2f}s | 工具调用: {self.tool_calls}次")

# 使用回调
callback = AgentMetricsCallback()
agent = create_agent(llm, tools)
# create_agent 创建的是一个 LangGraph StateGraph（状态图)
# StateGraph（整个 Agent，作为 Runnable 执行）
# │
# ├── 节点: model（模型调用节点）
# │     内部执行: llm.bind_tools(tools).invoke(messages)
# │     → 触发 on_chain_start / on_chain_end（bind_tools 返回的是 RunnableSequence）
# │     → 触发 on_llm_start / on_llm_end
# │     → LLM 决定是否调用工具
# │
# ├── 节点: tools（工具执行节点）
# │     内部执行: 逐个执行 LLM 要求调用的工具
# │     → 触发 on_tool_start / on_tool_end
# │
# └── 循环: 如果 LLM 输出中仍包含 tool_calls，
#           则回到 model 节点继续下一轮；
#           否则结束，返回最终消息。
result = agent.invoke(
    {"messages": [("user", "搜索最新AI新闻并总结")]},
    config={"callbacks": [callback]}
)