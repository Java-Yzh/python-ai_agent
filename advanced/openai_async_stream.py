"""
学习目标：异步调用 OpenAI API + 流式输出（用户体验的核心）
运行方式：python openai_async_stream.py
依赖：pip install openai python-dotenv
"""
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletionUserMessageParam
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ============================================================
# 1. 异步并行调用多个模型（Agent 对比场景）
# ============================================================
async def parallel_models_demo() -> None:
    """
    同时向两个模型（或同一模型的不同配置）发送请求。
    场景：对比不同模型的能力，同一个问题同时发给两个模型
    """
    async def ask_deepseek_flash(prompt: str, label: str) -> str:
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_FLASH"),
            messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
            max_tokens=2000,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.total_tokens
        return f"[{label}] {content}" + f" (Usage: {usage})"

    async def ask_deepseek_pro(prompt: str, label: str) -> str:
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_PRO"),
            messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
            max_tokens=2000,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.total_tokens
        return f"[{label}] {content}" + f" (Usage: {usage})"

    # 并行执行两个请求
    prompt = "请用三句话介绍 OpenAI 的历史"
    results = await asyncio.gather(
        ask_deepseek_flash(prompt, "deepseek-flash"),
        ask_deepseek_pro(prompt, "deepseek-pro"),
    )
    for r in results:
        print(r)

# ============================================================
# 2. 流式输出 —— 像 ChatGPT 一样逐字显示
# ============================================================
async def streaming_chat_demo() -> None:
    """
    流式输出：Server-Sent Events 模式。
    Agent 必须用流式输出才能提供好的用户体验，
    否则用户要等 5-10 秒才能看到完整回答。
    """
    print("流式输出示例：")

    stream = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[ChatCompletionUserMessageParam(role="user", content="请用三句话介绍 OpenAI 的历史")],
        stream=True,  # 开启流式
    )

    # 逐块处理
    async for chunk in stream:
        # delta.content 是本次新增的文本片段
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")  # 结束后换行


# ============================================================
# 3. 使用 tenacity 做智能重试（Agent 生产必备）
# ============================================================
@retry(
    # 最多重试 3 次后放弃，避免无限循环
    stop=stop_after_attempt(3),
    # 指数退避等待：第 1 次等 2s，第 2 次等 4s，第 3 次等 8s，上限 30s
    wait=wait_exponential(multiplier=1, min=2, max=30),
    # 只对可恢复的临时性错误重试：限流 429、网络连接失败、请求超时
    # 认证错误 401、参数错误 400 等不会重试，直接抛出
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    # 每次重试前自动打印警告日志，便于排查线上问题
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def chat_with_retry(prompt: str) -> str:
    """
    带自动重试的 OpenAI API 调用。
    - 遇到限流（429）、网络超时、连接错误时自动重试
    - 指数退避：第 1 次等 2 秒，第 2 次等 4 秒，第 3 次等 8 秒（最大 30 秒）
    - 遇到认证错误（401）等不可重试的错误则直接抛出
    """
    msg: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[msg],
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    print("=== 带重试的 OpenAI API 调用 ===")
    try:
        result = await chat_with_retry("用一句话介绍什么是限流（Rate Limiting）")
        print(f"回答：{result}")
    except Exception as e:
        print(f"重试 3 次后仍然失败：{e}")


if __name__ == "__main__":
    # 异步并行调用多个模型
    # asyncio.run(parallel_models_demo())

    # 流式输出
    # asyncio.run(streaming_chat_demo())

    # 带重试机制
    asyncio.run(main())