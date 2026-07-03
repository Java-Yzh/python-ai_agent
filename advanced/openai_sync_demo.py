"""
学习目标：掌握 OpenAI SDK 的同步调用方式，理解 Chat Completions API
运行前提：在 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL
运行方式：python openai_sync_demo.py
依赖：pip install openai python-dotenv
"""
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
import requests

# 加载 .env 文件（默认查找当前目录的 .env）
load_dotenv()

# ===================================================
# 初始化客户端
# base_url 可用于切换到兼容 OpenAI 格式的其他服务
# （如本地 Ollama、DeepSeek、智谱、阿里百炼等）
# ===================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# ============================================================
# 1. 最简单的对话调用
# ============================================================
def simple_caht() -> str:
    # 文档字符串，是 Python 的"正式文档机制"，
    # 写好后 IDE、help()、Sphinx 文档生成工具都能自动读取和展示，是 Python 社区推荐的函数/模块文档方式。
    # 你可以在 Python 终端里这样访问：help(simple_caht)
    # 会输出：
    # Help on function simple_caht in module __main__:
    #
    # simple_caht() -> str
    #     最基础的对话请求。返回 assistant 的回复文本。
    # 或者直接读取：
    # print(simple_caht.__doc__)
    # 会输出
    # 最基础的对话请求。返回 assistant 的回复文本。
    """最基础的对话请求。返回 assistant 的回复文本。"""
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        # 注意：较新版本的 OpenAI SDK（v1.0+）引入了严格类型检查机制
        # 以下写法会提示错误：
        # messages=[
        #     {"role": "system", "content": "你是一个简洁的助手，回答不超过30字。"},
        #     {"role": "user", "content": "什么是 Agent？一句话回答。"}
        # ],
        messages=[
            ChatCompletionSystemMessageParam(role="system", content="你是一个简洁的助手，回答不超过30字。"),
            ChatCompletionUserMessageParam(role="user", content="什么是 Agent？一句话回答。")
        ],
        # 温度：0-1，越小越 deterministic
        temperature=0.7,
        # 就是限制回答的最大长度，防止模型说太多
        max_tokens=100,
    )
    # 访问返回内容的标准方式
    return response.choices[0].message.content or ""

# ============================================================
# 2. 带工具调用（Function Calling）—— Agent 的核心
# ============================================================
def chat_with_tools(user_input: str) -> None:
    """
    演示 Function Calling。
    这是 Agent 能够"行动"的关键——LLM 不直接回答问题，
    而是返回一个工具调用请求，由我们的代码去执行。
    """

    def get_weather(city: str) -> str:
        """获取指定城市的实时天气"""
        # 这里替换为真实的天气 API 调用，示例用 wttr.in（免费、无需 key）
        try:
            resp = requests.get(f"https://wttr.in/{city}?format=%l:+%c+%t+%h+%w&lang=zh", timeout=10)
            resp.raise_for_status()
            return resp.text.strip()
        except Exception as e:
            return f"获取天气失败: {e}"

    # 工具注册表：name -> 实际函数
    tool_registry: dict[str, callable] = {
        "get_weather": get_weather,
    }

    # 工具 Schema 定义（与注册表对应）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如 Beijing、Tokyo",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            ChatCompletionSystemMessageParam(role="system", content="你是一个简洁的助手，回答不超过30字。"),
            ChatCompletionUserMessageParam(role="user", content=user_input),
        ],
        tools=tools,
        tool_choice="auto",
        # 提示：如果你的模型不支持 Function Calling，
        # 可以将 tool_choice 设置为 "none"，
        # 或者将 tool_choice 设置为 "function"，并指定一个函数名，
        # 模型将尝试调用该函数，但可能会失败。
        # 函数调用失败时，模型将返回一个错误消息。
        # 函数调用成功时，模型将返回一个 JSON 对象，
    )

    print("------------------------------------------")
    print("response:" + response.model_dump_json(indent=2))
    # 输出：
    # response: {
    #     "id": "0beb9e8b-367e-4ed0-af5a-6b37b3003e8f",
    #     "choices": [
    #         {
    #             "finish_reason": "tool_calls",
    #             "index": 0,
    #             "logprobs": null,
    #             "message": {
    #                 "content": "",
    #                 "refusal": null,
    #                 "role": "assistant",
    #                 "annotations": null,
    #                 "audio": null,
    #                 "function_call": null,
    #                 "tool_calls": [
    #                     {
    #                         "id": "call_00_HHKfoYRv44MCifaBdREe1856",
    #                         "function": {
    #                             "arguments": "{\"city\": \"济南\"}",
    #                             "name": "get_weather"
    #                         },
    #                         "type": "function",
    #                         "index": 0
    #                     }
    #                 ],
    #                 "reasoning_content": "用户想知道济南的天气。我需要调用get_weather工具来获取济南的天气信息。"
    #             }
    #         }
    #     ],
    #     "created": 1783046208,
    #     "model": "deepseek-v4-flash",
    #     "object": "chat.completion",
    #     "moderation": null,
    #     "service_tier": null,
    #     "system_fingerprint": "fp_8b330d02d0_prod0820_fp8_kvcache_20260402",
    #     "usage": {
    #         "completion_tokens": 64,
    #         "prompt_tokens": 299,
    #         "total_tokens": 363,
    #         "completion_tokens_details": {
    #             "accepted_prediction_tokens": null,
    #             "audio_tokens": null,
    #             "reasoning_tokens": 19,
    #             "rejected_prediction_tokens": null
    #         },
    #         "prompt_tokens_details": {
    #             "audio_tokens": null,
    #             "cached_tokens": 256
    #         },
    #         "prompt_cache_hit_tokens": 256,
    #         "prompt_cache_miss_tokens": 43
    #     }
    # }

    # ============================================================
    # OpenAI Chat Completions 响应结构（response）各字段含义：
    #
    # response.id          — 本次请求的唯一标识 ID
    #                        例："chatcmpl-abc123"
    #
    # response.object      — 对象类型，固定为 "chat.completion"
    #
    # response.created     — 请求创建时间，Unix 时间戳（秒）
    #
    # response.model       — 实际使用的模型名称
    #                        （可能与请求时指定的不同，如 gpt-4o-mini-2024-07-18）
    #
    # response.choices     — 返回结果列表，通常只有一个元素（choices[0]）
    #                        设置 n > 1 时会有多个，表示模型生成了多个候选回复。
    #
    # response.choices[0].index         — 结果索引，从 0 开始
    #
    # response.choices[0].message       — 模型的回复消息
    #   .message.role                   — 固定为 "assistant"
    #   .message.content                — 模型的文本回复
    #                                     正常回答时为字符串，调用工具时为 None
    #   .message.tool_calls             — 模型请求调用的工具列表
    #                                     正常回答时为 None，调用工具时为列表
    #     .tool_calls[i].id             — 本次工具调用的唯一 ID，
    #                                     回传 tool 消息时必须携带此 ID
    #     .tool_calls[i].type           — 固定为 "function"
    #     .tool_calls[i].function.name  — 模型要求调用的函数名
    #     .tool_calls[i].function.arguments — 函数参数，JSON 字符串格式
    #                                     需用 json.loads() 解析后使用
    #
    # response.choices[0].finish_reason — 生成结束的原因
    #   "stop"        — 模型自然结束（正常回答完毕）
    #   "tool_calls"  — 模型需要调用工具（此时 content 为 None）
    #   "length"      — 达到了 max_tokens 限制，回答被截断
    #   "content_filter" — 内容被安全过滤器拦截
    #
    # response.usage                — token 用量统计
    #   .usage.prompt_tokens        — 输入（prompt）消耗的 token 数
    #   .usage.completion_tokens    — 输出（回复）消耗的 token 数
    #   .usage.total_tokens         — 总 token 数 = prompt + completion
    #
    # 打印完整响应的推荐方式：
    #   print(response.model_dump_json(indent=2))
    #   注意：不可用 "response:" + response，Pydantic 模型不能直接与字符串拼接。
    # ============================================================

    choice = response.choices[0]
    message = choice.message

    # 判断 LLM 是否要求调用工具
    if message.tool_calls:
        print(f"LLM 要求调用工具！")

        # 收集所有工具调用的结果，支持一次对话中多次工具调用
        tool_messages = []

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  工具名: {func_name}")
            print(f"  参数:   {func_args}")

            # 从注册表查找并执行真实函数
            func = tool_registry.get(func_name)
            if func:
                tool_result = func(**func_args)
            else:
                tool_result = f"错误：未找到工具 {func_name}"

            print("------------------------------------------")
            print(f"  工具名: {func_name}")
            print(f"  执行结果: {tool_result}")

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # 将完整的对话历史（含所有工具结果）发回给 LLM
        # ============================================================
        # messages 中 role 的可选值及含义：
        #
        # "system"    — 系统指令，设定助手的行为、人格和约束。
        #               模型不会直接回复它，但会据此调整回复风格。
        #               通常放在 messages 列表的第一条。
        #               例：{"role": "system", "content": "你是一个专业的天气助手"}
        #
        # "user"      — 用户消息，代表用户说的话。
        #               是模型需要回答的内容。
        #               例：{"role": "user", "content": "北京天气如何？"}
        #
        # "assistant" — 助手消息，代表模型之前的回复。
        #               用途一：多轮对话中提供历史上下文，让模型记住之前的对话。
        #               用途二：Function Calling 中携带 tool_calls，
        #               告诉模型"你之前请求调用了这些工具"。
        #               例：{"role": "assistant", "content": "北京今天晴天"}
        #               例：{"role": "assistant", "content": None, "tool_calls": [...]}
        #
        # "tool"      — 工具结果，将工具执行的返回值反馈给模型。
        #               必须搭配 tool_call_id 使用，与 assistant 消息中的
        #               tool_calls 一一对应，让模型知道哪个调用得到了什么结果。
        #               例：{"role": "tool", "tool_call_id": "xxx", "content": "晴天，25°C"}
        #
        # 一次完整的 Function Calling 对话流：
        #   system    → "你是一个天气助手"           （设定角色）
        #   user      → "北京天气如何？"             （用户提问）
        #   assistant → tool_calls: [get_weather]   （模型请求调用工具）
        #   tool      → "晴天，25°C"                （工具执行结果）
        #   assistant → "北京今天晴天，气温25°C。"    （模型最终回复）
        # ============================================================
        final_messages = [
            {"role": "system", "content": "你是一个简洁的助手，回答不超过30字。"},
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc for tc in message.tool_calls],
            },
            # 这是 Python 的解包（unpacking）操作符 *，
            # 用在列表里会把 tool_messages 中的元素逐个展开插入到当前位置。
            *tool_messages,
        ]

        final_response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=final_messages,
        )

        print("------------------------------------------")
        print("final_response:" + final_response.model_dump_json(indent=2))

        print("------------------------------------------")
        print(f"\n最终回复: {final_response.choices[0].message.content}")
    else:
        # LLM 认为不需要调用工具，直接回答
        print(f"直接回复: {message.content}")


if __name__ == "__main__":
    # help(simple_caht)
    # print(simple_caht.__doc__)
    # 简单对话示例
    # print(simple_caht())

    # 工具调用示例
    print(chat_with_tools("济南天气如何？"))