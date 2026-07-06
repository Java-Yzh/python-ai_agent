import os

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_FLASH"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=1024,
)

# 字符串输出解析器
str_parser = StrOutputParser()

# JSON 结构化解析
class MovieReview(BaseModel):
    """电影影评结构"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分（0-10）")
    summary: str = Field(description="一句话总结")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")

json_parser = JsonOutputParser(pydantic_object=MovieReview)

review_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业影评人。\n{format_instructions}"),
    ("human", "请为电影《{movie_name}》写一篇简短影评")
])

# 注入格式说明到 prompt
print("---------------------------------------------------------------------------------")
print(json_parser.get_format_instructions())
print("---------------------------------------------------------------------------------")
# 预先填充模板中的部分变量，把一些固定的、不需要每次都变的参数（如格式指令）提前注入，后续使用时只需关心真正变化的部分
review_template = review_template.partial(
    format_instructions=json_parser.get_format_instructions()
)

# 这是 LangChain 的 LCEL（LangChain Expression Language） 语法，用 |（管道符）把多个组件串成一条链（Chain）。
# 数据的流向是从左到右：review_template → llm → json_parser
# review_template：接收输入（如 {"movie_name": "星际穿越"}），填充模板，生成完整的 Prompt
# llm：把 Prompt 发给大模型，获取原始文本输出
# json_parser：把大模型返回的 JSON 字符串解析成 Python 字典（符合 MovieReview 结构）
review_chain_without_json_parser = review_template | llm
result = review_chain_without_json_parser.invoke({"movie_name": "星际穿越"})
print(result)
# 无json_parser得到的result就是AIMessage的结构
# content='{"title": "星际穿越", "rating": 9.0, "summary": "一部融合硬科幻与人性温情的史诗级太空探索电影。", "pros": ["视觉效果震撼，黑洞与时间膨胀场景令人叹为观止", "科学顾问基普·索恩确保了物理学的严谨性", "父女情感线打动人心，马修·麦康纳演技出色", "汉斯·季默的配乐恢弘而情感饱满"], "cons": ["部分科学概念对普通观众可能难以理解", "剧情前半段节奏较慢，设定交代略显冗长", "结尾情节转折过于戏剧化，某些逻辑细节有待推敲"]}' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 328, 'prompt_tokens': 441, 'total_tokens': 769, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 184, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 57}, 'model_provider': 'openai', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '123dd884-282d-4e6c-8407-cca84fc9e8ce', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019f3602-bbd1-77b1-b014-4f3233013a94-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 441, 'output_tokens': 328, 'total_tokens': 769, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 184}}

print("---------------------------------------------------------------------------------")

review_chain = review_template | llm | json_parser
result = review_chain.invoke({"movie_name": "星际穿越"})
print(result)
# 有json_parser
# json_parser的作用：
# 1.从 AIMessage 中提取 content 字段。
# 2.把 JSON 字符串解析成 Python 字典
# 3.用 MovieReview Pydantic 模型校验，检查字段是否存在、类型是否匹配
# {'title': '星际穿越', 'rating': 9.0, 'summary': '一场穿越虫洞与黑洞的壮丽星际之旅，探讨爱与时间的深刻主题。', 'pros': ['视觉效果震撼', '科学严谨', '情感动人', '配乐出色', '主题深刻'], 'cons': ['节奏稍慢', '部分科学概念复杂难懂', '结局略显理想化']}