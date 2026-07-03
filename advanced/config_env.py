"""
学习目标：使用 python-dotenv 管理敏感配置
运行方式：先在同目录创建 .env 文件，写入 OPENAI_API_KEY=sk-xxx，然后 python config_env.py
"""
from dotenv import load_dotenv
import os

# 加载 .env 文件（默认查找当前目录的 .env）
load_dotenv()

# ============================================================
# 核心模式：os.getenv("KEY", "默认值")
# - 第二个参数是找不到环境变量时的 fallback
# - 严禁在代码中直接写死 key
# ============================================================
api_key: str = os.getenv("OPENAI_API_KEY", "")
base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

# ============================================================
# if __name__ == "__main__": 详解
#
# 这不是语法规范，而是 Python 的一个语言机制，属于"惯用法"（idiom）。
#
# __name__ 是什么？
#   每个 Python 文件（模块）在运行时都有一个内置变量 __name__，
#   它的值取决于文件是如何被执行的：
#     - 直接运行该文件（python config_env.py）→ __name__ == "__main__"
#     - 被其他文件导入（import config_demo）    → __name__ == "config_demo"（即模块名）
#
# 这行代码的作用：
#   直接运行 python config_env.py → 条件为真，下面的打印代码会执行
#   被导入 import config_demo     → 条件为假，下面的打印代码不会执行
#
# 为什么要这样写？
#   核心目的：让一个文件既能被直接运行，又能被安全地导入复用。
#   - 第 5-19 行（加载环境变量、读取配置）在导入时就会执行，
#     其他文件可以直接 from config_demo import api_key 来使用这些变量。
#   - 第 22-26 行（打印调试信息）只在直接运行时执行，
#     不会在别人导入这个模块时产生意外的 print 输出。
#
# 就类似于Java中的main方法：public static void main(String[] args)
# 如果不加这个判断，别人一 import config_demo，控制台就会打印一堆调试信息，
# 这通常不是你想要的。
# ============================================================
if __name__ == "__main__":
    masked = api_key[:7] + "..." if api_key else "未配置"
    print(f"API Key:  {masked}")
    print(f"Base URL: {base_url}")
    print(f"Model:    {model}")
    print(f"Retries:  {max_retries}")