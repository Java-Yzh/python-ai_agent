# 加载文档 -> 分割 -> 嵌入 -> 存储到 向量数据库
import os
# HF_HUB_OFFLINE=1：告诉 huggingface_hub 只从本地缓存读取，不发起任何网络请求
# HF_ENDPOINT 仍保留作为后备，如果将来需要下载新模型时走镜像站
# 必须在所有 huggingface 相关 import 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import (
    PyPDFLoader,         # PDF
    TextLoader,          # 纯文本
    CSVLoader,           # CSV
    UnstructuredMarkdownLoader,  # Markdown 已废弃
    WebBaseLoader,       # 网页
)
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 加载文档
# 用 os.path.dirname(__file__) 取脚本自身所在目录，再拼上 doc.md，这样无论从哪个工作目录运行都能正确找到文件。
doc_path = os.path.join(os.path.dirname(__file__), "doc.md")
loader = TextLoader(doc_path)
pages = loader.load()
print(f"共 {len(pages)} 个文档块")

# 通用分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每块最大字符数
    chunk_overlap=20,  # 块间重叠字符数（保证语义连续性）
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 分割优先级
    length_function=len,
)

# 切割文档
documents = text_splitter.split_documents(pages)
print(f"分割后共 {len(documents)} 个文本块")

# 嵌入模型：HuggingFace 开源模型，完全本地运行
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # 有 GPU 改为 "cuda"
    encode_kwargs={"normalize_embeddings": True},
)

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_knowledge_base"
)

# 后续加载已有向量库（无需重复嵌入）
# loaded_vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings,
#     collection_name="my_knowledge_base"
# )