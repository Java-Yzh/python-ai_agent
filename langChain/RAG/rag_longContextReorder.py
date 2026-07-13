# LongContextReorder 检索策略（上下文重排序，解决"中间遗忘"问题）
import os
# HF_HUB_OFFLINE=1：告诉 huggingface_hub 只从本地缓存读取，不发起任何网络请求
# HF_ENDPOINT 仍保留作为后备，如果将来需要下载新模型时走镜像站
# 必须在所有 huggingface 相关 import 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder

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
# LongContextReorder（长上下文重排序）
# 背景：LLM 存在"中间遗忘"（Lost in the Middle）现象——
#   放在上下文开头和结尾的信息容易被记住，放在中间的容易被忽略
# 作用：检索后把最相关的文档放到首尾位置，较不相关的放中间
#   原始顺序：[doc1, doc2, doc3, doc4, doc5]（按相似度降序）
#   重排后：  [doc1, doc3, doc5, doc4, doc2]（最相关在首尾）
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
reorder = LongContextReorder()

# 用 RunnableLambda 包装 transform_documents，接入 LCEL 管道
# 流程：检索 5 条 → 重排序 → 格式化为上下文
reorder_retriever = retriever | RunnableLambda(lambda docs: reorder.transform_documents(docs))

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
#   1. 字典：  单个字符串 → {"context": 检索+重排+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": reorder_retriever | format_docs,    # 检索 + 重排 + 格式化
        "question": RunnablePassthrough()              # 问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试
answer = rag_chain.invoke("哆啦A梦的三件秘密道具是什么？")
print(answer)

# 直接查看检索结果（对比重排前后顺序）
print("=== 重排前（原始检索顺序）===")
raw_docs = retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(raw_docs, 1):
    print(f"  [{i}] {doc.page_content[:60]}...")

print("\n=== 重排后（首尾放最相关）===")
reordered_docs = reorder.transform_documents(raw_docs)
for i, doc in enumerate(reordered_docs, 1):
    print(f"  [{i}] {doc.page_content[:60]}...")
