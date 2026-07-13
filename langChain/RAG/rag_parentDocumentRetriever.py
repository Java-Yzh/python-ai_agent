# ParentDocumentRetriever 高级检索（子块检索、父块返回，兼顾召回率与上下文完整性）
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
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# === 加载原始文档 ===
doc_path = os.path.join(os.path.dirname(__file__), "doc.md")
loader = TextLoader(doc_path)
pages = loader.load()

# === ParentDocumentRetriever 核心思想 ===
# 普通检索的矛盾：chunk 太小 → 召回高但上下文不完整；chunk 太大 → 召回低但上下文完整
# ParentDocumentRetriever 的解决方案：两者兼得
#   1. 把文档切成大块（父块），再切成小块（子块）
#   2. 用子块做向量索引（小块语义集中 → 召回率高）
#   3. 命中子块后，返回其所属的父块（大块 → 上下文完整）
#
# 示意图：
#   父块（500字）：  ████████████████████
#   子块（100字）：  ████  ████  ████  ████
#   向量索引：       ↑子块1  ↑子块2  ↑子块3  ↑子块4
#   命中子块2 → 返回整个父块

# 父分割器：大块（保证返回的上下文完整）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 子分割器：小块（提高检索召回率）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 子块向量库（独立目录，避免与现有 chroma_db 冲突）
child_vectorstore = Chroma(
    persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db_parent"),
    embedding_function=embeddings,
    collection_name="parent_doc_child"
)

# 父块文档存储（内存中，key 为父块 ID，value 为父块原文）
docstore = InMemoryStore()

# 创建 ParentDocumentRetriever
parent_doc_retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加文档：自动按父分割器切大块，再按子分割器切小块，子块入向量库，父块入 docstore
parent_doc_retriever.add_documents(pages)

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
#   1. 字典：  单个字符串 → {"context": 父子检索+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": parent_doc_retriever | format_docs,    # 父子检索 + 格式化
        "question": RunnablePassthrough()                 # 问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试
answer = rag_chain.invoke("哆啦A梦的三件秘密道具是什么？")
print(answer)

# 直接查看检索结果
docs = parent_doc_retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()
