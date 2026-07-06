# 模型的输入与输出 基础的调用
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 多轮对话：手动管理消息列表
def chat_turn():
    messages = [
        SystemMessage(content="你是一个数学老师，请耐心讲解"),
        HumanMessage(content="什么是微积分？"),
        AIMessage(content="微积分是研究变化率与累积量的数学分支..."),
        HumanMessage(content="请举个例子")
    ]
    return llm.invoke(messages)

# 批量调用：一次调用，返回 AIMessageList
def chat_batch():
    questions = ["什么是Python？", "什么是LangChain？", "什么是向量数据库？"]
    responses = llm.batch(questions) # 并行处理，返回 AIMessageList
    # 批量调用时返回结果的顺序与输入问题的顺序是一致的
    for question, response in zip(questions, responses):
        print(f"问题：{question}")
        print(f"回答：{response.content}")

# 流式输出
def chat_stream():
    for chunk in llm.stream("用 200 字介绍深度学习"):
        # end=""：指定 print 输出后的结尾字符。默认是 "\n"（换行），
        # 这里设为空字符串，意味着不换行，让每次输出的内容紧挨着拼接在一起，实现流式输出的效果。
        # flush=True：强制立即刷新输出缓冲区。正常情况下 print 会先写入缓冲区，等缓冲区满或遇到换行才真正显示到终端。
        # 设为 True 后每个 chunk 都会立刻显示出来，用户可以看到逐字输出的效果，而不是等全部生成完才一次性显示。
        print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    # invoke：单次调用，返回 AIMessage
    # response = llm.invoke("用一句话解释 LangChain")
    # print(response.model_dump_json())
    # {
    #     "content": "LangChain 是一个开源框架，用于简化和模块化构建基于大语言模型（LLM）的应用程序，通过链式调用、数据连接和工具集成来管理复杂的工作流。",
    #     "additional_kwargs": {
    #         "refusal": null
    #     },
    #     "response_metadata": {
    #         "token_usage": {
    #             "completion_tokens": 88,
    #             "prompt_tokens": 9,
    #             "total_tokens": 97,
    #             "completion_tokens_details": {
    #                 "accepted_prediction_tokens": null,
    #                 "audio_tokens": null,
    #                 "reasoning_tokens": 47,
    #                 "rejected_prediction_tokens": null
    #             },
    #             "prompt_tokens_details": {
    #                 "audio_tokens": null,
    #                 "cached_tokens": 0
    #             },
    #             "prompt_cache_hit_tokens": 0,
    #             "prompt_cache_miss_tokens": 9
    #         },
    #         "model_provider": "openai",
    #         "model_name": "deepseek-v4-flash",
    #         "system_fingerprint": "fp_8b330d02d0_prod0820_fp8_kvcache_20260402",
    #         "id": "666114b3-379d-4397-b77a-c257b5ab0d93",
    #         "finish_reason": "stop",
    #         "logprobs": null
    #     },
    #     "type": "ai",
    #     "name": null,
    #     "id": "lc_run--019f3586-1141-7e73-a621-c76aed706aac-0",
    #     "tool_calls": [
    #
    #     ],
    #     "invalid_tool_calls": [
    #
    #     ],
    #     "usage_metadata": {
    #         "input_tokens": 9,
    #         "output_tokens": 88,
    #         "total_tokens": 97,
    #         "input_token_details": {
    #             "cache_read": 0
    #         },
    #         "output_token_details": {
    #             "reasoning": 47
    #         }
    #     }
    # }

    # 多轮对话示例
    # result_turn = chat_turn()
    # print(result_turn.content)

    # 批量调用
    # chat_batch()

    # 流式输出
    chat_stream()