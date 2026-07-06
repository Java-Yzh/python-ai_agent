# Prompt Template：提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 方式一：单轮对话
simple_template = ChatPromptTemplate.from_template(
    "将以下文本翻译成{target_language}：\n{text}"
)

# 填入变量 → 生成 Messages
prompt_value = simple_template.invoke({
    "target_language": "英文",
    "text": "人工智能正在改变世界"
})
print(prompt_value.messages)
# [HumanMessage(content='将以下文本翻译成英文：\n人工智能正在改变世界', additional_kwargs={}, response_metadata={})]

from langchain_core.prompts import MessagesPlaceholder

# 方式二：多轮对话
multi_turn = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，回答风格：{style}"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位
    ("human", "{question}")
])

prompt_value_multi = multi_turn.invoke({
    "role": "Python 技术专家",
    "style": "简洁、附带代码示例",
    "history": [],  # 初始无历史
    "question": "Python 中如何处理 JSON？"
})
print(prompt_value_multi.messages)