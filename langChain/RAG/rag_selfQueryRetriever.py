# SelfQueryRetriever 高级检索（自然语言转结构化查询，支持 metadata 过滤）
import os
# HF_HUB_OFFLINE=1：告诉 huggingface_hub 只从本地缓存读取，不发起任何网络请求
# HF_ENDPOINT 仍保留作为后备，如果将来需要下载新模型时走镜像站
# 必须在所有 huggingface 相关 import 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv

# === 兼容性修复 ===
# langchain_community 0.4.2 移除了部分 vectorstore 集成（DatabricksVectorSearch 等），
# 但 langchain_classic 1.0.8 的 SelfQueryRetriever 仍在一个 bulk import 中尝试导入它们，
# 导致 ImportError。此处注入占位类使导入不报错，不影响实际使用的 Chroma 向量库。
import langchain_community.vectorstores as _vs
for _name in ['DatabricksVectorSearch', 'DeepLake', 'Milvus', 'Neo4jVector', 'Qdrant', 'Weaviate', 'MongoDBAtlasVectorSearch', 'Pinecone']:
    if not hasattr(_vs, _name):
        setattr(_vs, _name, type(_name, (), {}))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.retrievers.self_query.base import AttributeInfo

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

# === SelfQueryRetriever 核心思想 ===
# 普通检索只能做语义匹配，无法利用文档的 metadata 做过滤
# SelfQueryRetriever 用 LLM 把自然语言拆解为两部分：
#   1. 语义查询（用于向量相似度检索）
#   2. 结构化过滤条件（用于 metadata 精确过滤）
#
# 示例：
#   用户问："来源是 doc.md 的关于哆啦A梦秘密道具的文档"
#   LLM 拆解：
#     语义查询 = "哆啦A梦秘密道具"
#     过滤条件 = source == "doc.md"
#   最终：在 source=doc.md 的文档中做向量检索
#
# 注意：此检索器依赖 LLM 的函数调用（function calling）能力
#   DeepSeek API 兼容 OpenAI 接口，支持 function calling

# 定义文档的 metadata 字段信息（LLM 需要知道有哪些字段可过滤）
# 当前文档只有 source 字段（TextLoader 自动添加的文件路径）
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="文档来源文件路径，例如 /Users/bmht/Code/Python/my_first_project/langChain/RAG/doc.md",
        type="string",
    ),
]

# 创建 SelfQueryRetriever
# document_contents：告诉 LLM 文档库的大致内容，帮助它理解查询意图
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="哆啦A梦与超级赛亚人跨界合作的故事",
    metadata_field_info=metadata_field_info,
    enable_limit=True,   # 允许 LLM 在查询中指定返回数量（如"给我3条..."）
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
#   1. 字典：  单个字符串 → {"context": 自查询检索+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": self_query_retriever | format_docs,    # 自查询检索 + 格式化
        "question": RunnablePassthrough()                 # 问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试：纯语义查询（无 metadata 过滤）
answer = rag_chain.invoke("哆啦A梦的三件秘密道具是什么？")
print(answer)

# 直接查看检索结果
# 测试1：纯语义检索
print("=== 测试1：纯语义检索 ===")
docs = self_query_retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()

# 测试2：带 metadata 过滤的查询
# LLM 会自动解析出过滤条件，只返回 source 为 doc.md 的文档
print("=== 测试2：带 metadata 过滤 ===")
docs2 = self_query_retriever.invoke("来源是/Users/bmht/Code/Python/my_first_project/langChain/RAG/doc.md的关于哆啦A梦秘密道具的文档")
for i, doc in enumerate(docs2, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()
