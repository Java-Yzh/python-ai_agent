import os
import json
import ast
import operator

from dotenv import load_dotenv
from openai import OpenAI
from langchain_tavily import TavilySearch

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

MODEL = os.getenv("DEEPSEEK_MODEL_FLASH")

# ── 1. 定义工具 ──────────────────────────────

search = TavilySearch(max_results=3)

def calculator(expression: str) -> str:
    try:
        allowed_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }

        def safe_eval(node):
            if isinstance(node, ast.Expression):
                return safe_eval(node.body)
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](safe_eval(node.operand))
            elif isinstance(node, ast.Constant):
                return node.value
            raise ValueError(f"不支持的操作: {type(node)}")

        tree = ast.parse(expression, mode='eval')
        return str(safe_eval(tree))
    except Exception as e:
        return f"计算错误: {e}"

# 工具注册表：名称 → (描述, 执行函数)
TOOL_REGISTRY = {
    "tavily_search": {
        "description": "搜索互联网获取最新信息",
        "parameters": {"query": "搜索关键词字符串"},
        "func": lambda query: search.invoke(query),
    },
    "calculator": {
        "description": "执行数学计算",
        "parameters": {"expression": "数学表达式字符串，如 '2 + 3 * 4'"},
        "func": calculator,
    },
}

# 转为 OpenAI function calling 格式
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in info["parameters"].items()},
                "required": list(info["parameters"].keys()),
            },
        },
    }
    for name, info in TOOL_REGISTRY.items()
]

# ── 2. 执行工具调用 ────────────────────────────

def execute_tool_call(name: str, arguments: dict) -> str:
    if name not in TOOL_REGISTRY:
        return f"未知工具: {name}"
    func = TOOL_REGISTRY[name]["func"]
    return func(**arguments)

# ── 3. ReAct 循环（核心） ─────────────────────
# 每轮循环：调用 LLM → 将响应加入历史 → 检查是否有 tool_calls
# 有 tool_calls → 逐个执行工具，把结果加入历史，进入下一轮
# 无 tool_calls → content 即为最终回答，直接结束并输出即可
def react_agent(user_input: str, max_steps: int = 10) -> None:
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以搜索信息和进行数学计算。"},
        {"role": "user", "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'='*20} 第 {step} 轮 {'='*20}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        # 把 LLM 的响应加入历史
        # model_dump() 之后 — 普通 dict
        # {
        #     "role": "assistant",
        #     "content": None,
        #     "tool_calls": [{"id": "...", "function": {"name": "tavily_search", "arguments": "{...}"}}],
        # }
        messages.append(msg.model_dump())

        # 如果 LLM 没有调工具 → 结束
        if not msg.tool_calls:
            print(f"[最终回答] {msg.content}")
            return

        # 有 tool_calls → 逐个执行，把结果加入历史
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"[调用工具] {name}({arguments})")

            result = str(execute_tool_call(name, arguments))
            print(f"[工具结果] {result[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    print(f"\n达到最大步数 {max_steps}，强制结束")

# ── 4. 运行 ──────────────────────────────────

if __name__ == "__main__":
    react_agent("2024年诺贝尔物理学奖得主是谁？把他的年龄除以 2 是多少？")
