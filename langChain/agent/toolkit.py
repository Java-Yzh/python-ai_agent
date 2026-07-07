# 工具包toolkit示例

import os
from langchain_openai import ChatOpenAI
# SQL 数据库工具包
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 连接到 SQLite 数据库
db = SQLDatabase.from_uri("sqlite:///chinook.db")  # 示例数据库

# 创建工具包：自动生成查询、描述表结构等工具
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()

# 查看自动生成的工具
for t in sql_tools:
    print(f"{t.name}: {t.description[:80]}...")
    # print(t)