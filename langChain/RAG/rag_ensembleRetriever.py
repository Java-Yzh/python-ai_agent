# EnsembleRetriever 高级检索（混合检索：向量语义 + BM25 关键词）
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
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
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

# === 加载原始文档，为 BM25 准备数据 ===
# BM25 是基于关键词的稀疏检索，不需要嵌入向量，但需要原始文本
doc_path = os.path.join(os.path.dirname(__file__), "doc.md")
loader = TextLoader(doc_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""],
    length_function=len,
)
documents = text_splitter.split_documents(pages)

# === 创建两路检索器 ===
# 路线 A：向量检索（语义匹配，擅长理解同义词、近义词）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 路线 B：BM25 检索（关键词匹配，擅长精确匹配专有名词）
# BM25 是 TF-IDF 的改进版，考虑词频饱和和文档长度归一化
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4

# === 融合检索器 ===
# EnsembleRetriever 使用 RRF（Reciprocal Rank Fusion）算法融合两路结果：
#   - 向量检索擅长语义相似（"秘密道具" ≈ "隐藏武器"）
#   - BM25 擅长关键词精确匹配（"哆啦A梦" 作为专有名词）
#   - 两者互补，中文场景效果显著优于单一检索
# weights 控制两路权重：[向量权重, BM25权重]
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
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
#   1. 字典：  单个字符串 → {"context": 混合检索+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": ensemble_retriever | format_docs,     # 混合检索 + 格式化
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
docs = ensemble_retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()
