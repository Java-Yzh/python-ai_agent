# MMR 检索策略（最大边际相关性，提升结果多样性）
import os
# HF_HUB_OFFLINE=1：告诉 huggingface_hub 只从本地缓存读取，不发起任何网络请求
# HF_ENDPOINT 仍保留作为后备，如果将来需要下载新模型时走镜像站
# 必须在所有 huggingface 相关 import 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 加载已有数据库
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma(
    persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"),
    embedding_function=embeddings,
    collection_name="my_knowledge_base"
)
# MMR（Maximal Marginal Relevance）检索策略
# 与普通相似度检索的区别：
#   - 普通检索：只看查询与文档的相似度，可能返回内容高度重复的 N 条
#   - MMR 检索：在相似度和多样性之间做平衡，先召回 fetch_k 条，再从中选 k 条
#     使得选出的文档既相关又彼此不重复
#
# 参数说明：
#   k=5            最终返回 5 条文档
#   fetch_k=20     先召回 20 条候选（从这 20 条中做 MMR 选择）
#   lambda_mult=0.5 平衡因子：1.0=纯相关性（退化成普通检索），0.0=纯多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# RAG Prompt 模板
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的知识助手。根据以下上下文回答问题。
如果上下文中没有相关信息，请如实说"我不确定"。

上下文：
{context}"""),
    ("human", "{question}")
])

# 格式化检索到的文档
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

# 完整 RAG 链（LCEL 管道，invoke 只需传入一个问题字符串，四步自动完成）
#   1. 字典：  单个字符串 → {"context": 检索+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": retriever | format_docs,     # 检索 + 格式化
        "question": RunnablePassthrough()        # 问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试
answer = rag_chain.invoke("哆啦A梦的三件秘密道具是什么？")
print(answer)

# 直接查看检索结果
docs = retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()
