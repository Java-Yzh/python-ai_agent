# MultiVectorRetriever 高级检索（多向量索引：用摘要做检索，返回原文档）
import os
import uuid
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
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(pages)

# === MultiVectorRetriever 核心思想 ===
# 普通检索：直接用原文做向量索引，长文档语义分散 → 检索不精准
# MultiVectorRetriever：为每个文档生成多种向量表示（摘要 / 假设问题），用它们做索引
#   检索时：用户问题匹配"摘要向量" → 命中后从 docstore 取回"完整原文"
#
# 本例使用 LLM 生成摘要作为索引内容：
#   原文（500字）：哆啦A梦带着特兰克斯与大雄启动时光机，穿越到了...
#   摘要（50字）：哆啦A梦一行人穿越到未来世界对抗黑暗赛亚人
#   向量索引：用摘要做嵌入（语义集中 → 检索精准）
#   docstore：存原文（返回时给 LLM 完整上下文）

# 摘要向量库（独立目录，避免与现有 chroma_db 冲突）
summary_vectorstore = Chroma(
    persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db_multivector"),
    embedding_function=embeddings,
    collection_name="multi_vector_summary"
)

# 原文文档存储（内存中，key 为 doc_id，value 为原文 Document）
docstore = InMemoryStore()

# 创建 MultiVectorRetriever
# id_key：指定 metadata 中哪个字段作为 docstore 的关联键
multi_vector_retriever = MultiVectorRetriever(
    vectorstore=summary_vectorstore,
    docstore=docstore,
    id_key="doc_id",
)

# === 用 LLM 为每个文档块生成摘要，建立多向量索引 ===
# 摘要 prompt：提炼关键信息，让向量索引更精准
summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "请为以下文档内容生成一段简洁摘要，突出关键信息，不超过50字。"),
        ("human", "{content}")
    ])
    | llm
    | StrOutputParser()
)

print(f"正在为 {len(documents)} 个文档块生成摘要...")
for doc in documents:
    doc_id = str(uuid.uuid4())
    # 用 LLM 生成摘要
    summary = summary_chain.invoke({"content": doc.page_content})
    # 原文存入 docstore（检索命中摘要后，从这里取回原文）
    docstore.put([doc_id], [doc])
    # 摘要存入向量库（metadata 中带 doc_id 用于关联）
    summary_doc = Document(page_content=summary, metadata={"doc_id": doc_id})
    summary_vectorstore.add_documents([summary_doc])
print("摘要生成完成！\n")

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
#   1. 字典：  单个字符串 → {"context": 多向量检索+格式化后的文档, "question": 原样透传}
#   2. rag_prompt：用 context + question 填充模板，生成完整 prompt
#   3. llm：   将完整 prompt 发给大模型，返回回复消息
#   4. StrOutputParser()：从回复消息中提取纯文本字符串
rag_chain = (
    {
        "context": multi_vector_retriever | format_docs,    # 多向量检索 + 格式化
        "question": RunnablePassthrough()                    # 问题透传
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 测试
answer = rag_chain.invoke("哆啦A梦的三件秘密道具是什么？")
print(answer)

# 直接查看检索结果（返回的是原文而非摘要）
docs = multi_vector_retriever.invoke("哆啦A梦的三件秘密道具是什么？")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content}")
    print()
