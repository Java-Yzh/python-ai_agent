# LCEL = LangChain Expression Language
# 核心：用管道符 | 串联组件，构成 RunnableSequence（执行链）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch, RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 一个简单的完整的 RunnableSequence（链）
def simple_chain():
    chain = (
        ChatPromptTemplate.from_template("用{language}解释：{concept}")
        | llm
        | StrOutputParser()
    )
    print(chain.invoke({"language": "中文", "concept": "链式调用"}))

    # 等同于：
    # prompt = ChatPromptTemplate.from_template("...")
    # prompt_value = prompt.invoke({"language": "Python", "concept": "装饰器"})
    # response = llm.invoke(prompt_value)         # → AIMessage
    # result = StrOutputParser().invoke(response)  # → str

# 稍微复杂的 链
def complex_chain():
    # 链1：大纲生成
    outline_prompt = ChatPromptTemplate.from_template(
        "为「{topic}」生成一份博客大纲，包含 4-6 个要点"
    )
    outline_chain = outline_prompt | llm | StrOutputParser()

    # 链2：内容生成
    body_prompt = ChatPromptTemplate.from_template(
        "根据以下大纲，写一篇 800 字博客文章：\n{outline}"
    )
    body_chain = body_prompt | llm | StrOutputParser()

    # format_body_input 是一个中间转换函数，用于将链 1 的输出适配为链 2 的输入。
    # 具体来说：
    # 链 1（outline_chain）的输出是一个 str（博客大纲文本）
    # 链 2（body_chain）的 prompt 模板需要一个 {"outline": ...} 的字典作为输入
    # 这个函数的作用就是把链 1 输出的纯字符串包装成链 2 所需的 dict 格式。
    def format_body_input(outline: str) -> dict:
        """将链1输出转换为链2的输入格式"""
        return {"outline": outline}

    full_chain = (
            outline_chain
            | RunnableLambda(format_body_input)
            | body_chain
    )

    result = full_chain.invoke({"topic": "如何用 LangChain 构建一个完整的链式调用"})
    print(result[:200])

# 多个链并行执行，parallel：并行的
def parallel_chain():
    # 同时生成：标题、摘要、关键词
    title_prompt = ChatPromptTemplate.from_template("为以下文章生成一个标题：\n{article}")
    summary_prompt = ChatPromptTemplate.from_template("为以下文章写一个 50 字摘要：\n{article}")
    keywords_prompt = ChatPromptTemplate.from_template("为以下文章提取 5 个关键词：\n{article}")

    chain = RunnableParallel(
        title=title_prompt | llm | StrOutputParser(),
        summary=summary_prompt | llm | StrOutputParser(),
        keywords=keywords_prompt | llm | StrOutputParser(),
    )
    result = chain.invoke({"article": "近日一则老年用户遭遇网络营销骚扰的报道引发关注。88 岁奶奶手机有 77 万多条未读消息，"
                             "多来自企业微信群。这些群由奶奶添加的企业微信 联系人直接拉入，40 人以下群无需本人同意，目的是向老人卖保健品等。"
                             "奶奶三年添加 1200 个企微账号，近一年被拉入 1900 多个群。家人将微信切换到「未成年人模式」暂缓失控，微信承诺下一步对异常企业微信上线拉群强制确认功能。"})
    print(result["title"])
    print(result["summary"])
    print(result["keywords"])

# 带条件的链
def conditional_chain():
    # 根据不同情绪给出不同回复风格
    positive_chain = ChatPromptTemplate.from_template(
        "用户很满意！热情地回复：{input}"
    ) | llm | StrOutputParser()

    negative_chain = ChatPromptTemplate.from_template(
        "用户不满！道歉并给出补偿方案：{input}"
    ) | llm | StrOutputParser()

    neutral_chain = ChatPromptTemplate.from_template(
        "正常回复：{input}"
    ) | llm | StrOutputParser()

    # 分支路由
    branch = RunnableBranch(
        (lambda x: "满意" in x["input"], positive_chain),
        (lambda x: "不满" in x["input"] or "投诉" in x["input"], negative_chain),
        neutral_chain  # 默认分支
    )

    print(branch.invoke({"input": "我对你们的产品非常满意！"}))
    print(branch.invoke({"input": "你们的服务太差了，我要投诉！"}))


# 数据透传
# RunnablePassthrough 的典型场景是：部分数据需要处理，其余数据原样保留，最终两者在下游合并使用
def passthrough_chain():
    # 场景：用户问题先做意图分析，原始问题原样保留，最终合并生成回答

    intent_chain = (
            ChatPromptTemplate.from_template("判断以下用户问题的意图类别（咨询/投诉/建议）：{question}")
            | llm
            | StrOutputParser()
    )

    full_chain = (
        RunnableParallel(
            intent=intent_chain,
            question=RunnablePassthrough()
        )
        | RunnableLambda(lambda x: {"intent": x["intent"], "question": x["question"]})
        | ChatPromptTemplate.from_template(
            "用户问题：{question}\n意图类别：{intent}\n请根据意图给出合适的回复。要附带说明用户的问题和其意图类别。"
        )
        | llm
        | StrOutputParser()
    )

    result = full_chain.invoke({"question": "你们的 App 加载速度太慢了，能不能优化一下？"})
    print(result)

if __name__ == "__main__":
    # 简单基础链
    # simple_chain()

    # 稍微复杂的链
    # complex_chain()

    # 并行链
    # parallel_chain()

    # 条件链
    # conditional_chain()

    # 数据透传
    passthrough_chain()