# 创建供Agent使用的tool

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
from langchain_core.tools import StructuredTool

# ========== 关于 Schema ==========
# Schema（模式）是对数据结构的形式化描述，这里指 JSON Schema —— 一种用来描述 JSON 数据长什么样的规范。
#
# 这个 schema 不是给人看的，而是给 LLM（大模型）看的 —— 它是 LLM 和你的工具之间的"接口文档"。
#
# 当 Agent 工作时，流程如下：
#   用户: "帮我查一下北京的天气"
#     ↓
#   Agent（LLM）: 我手头有哪些工具可以用？
#     ↓
#   系统把 schema 列表给 LLM 看:
#     → get_weather: 获取指定城市的实时天气，参数: city(string, 必填)
#     ↓
#   LLM 理解后决定: 调用 get_weather，传参 city="北京"
#     ↓
#   函数执行，返回结果
#
# LLM 通过读 schema 来知道：有哪些工具可用、每个工具干什么、需要传什么参数、什么类型、哪些必填。
# 没有 schema，LLM 就不知道该怎么调用你的工具。
#
# LangChain 的 @tool 装饰器会自动从函数签名中推断 schema：
#   - 函数名 get_weather        → 工具名
#   - docstring                 → 工具描述
#   - city: str（无默认值）      → 参数名 city，类型 string，必填
#
# 自动生成的 schema 如下：
# {
#   "name": "get_weather",
#   "description": "获取指定城市的实时天气",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "city": {
#         "type": "string",
#         "description": "city"
#       }
#     },
#     "required": ["city"]
#   }
# }
#
# 类比：就像 FastAPI 根据函数参数注解自动生成 API 文档（Swagger），
# @tool 根据函数签名自动生成给 LLM 看的"工具文档"（schema）。
# ================================

# 基础用法：自动推断参数 schema
# LangChain 的 @tool 装饰器能够根据你的函数签名（类型注解和 docstring）自动生成工具参数的 JSON Schema，而不需要手动定义。
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气"""
    # 实际项目中替换为真实 API 调用
    weather_data = {
        "北京": "晴天，25°C，湿度45%",
        "上海": "多云，28°C，湿度60%",
        "深圳": "阵雨，30°C，湿度80%"
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

# 高级用法：自定义schema
class OrderQueryInput(BaseModel):
    """订单查询参数"""
    order_id: str = Field(description="订单号，格式为 ORD-2024-XXXX")
    include_details: bool = Field(
        default=False,
        description="是否包含商品详情"
    )

@tool(args_schema=OrderQueryInput)
def query_order(order_id: str, include_details: bool = False) -> str:
    """查询订单状态和详情"""
    orders = {
        "ORD-2024-0001": {
            "status": "已发货", "amount": 299.00,
            "items": ["Python 编程书", "机械键盘"]
        },
        "ORD-2024-0002": {
            "status": "待付款", "amount": 1599.00,
            "items": ["显示器"]
        }
    }
    order = orders.get(order_id, {})
    if not order:
        return f"未找到订单 {order_id}"

    base = f"订单 {order_id}：状态={order['status']}，金额=¥{order['amount']}"
    if include_details:
        base += f"，商品={order['items']}"
    return base

# 从函数创建tool
# ========== @tool 装饰器 vs StructuredTool.from_function() ==========
# 两种方式功能上基本等价，核心区别在于"控制权"：
#
# @tool 装饰器：
#   - 写法：更简洁，直接加在函数上面
#   - 工具名：自动取函数名
#   - 描述：自动取 docstring
#   - 适用场景：函数自己写的，可以直接改
#
# StructuredTool.from_function()：
#   - 写法：稍繁琐，需要额外调用
#   - 工具名：可以自定义 name
#   - 描述：可以自定义 description，覆盖 docstring
#   - 适用场景：函数别人写的或已有的，不方便加装饰器
#
# StructuredTool.from_function() 的典型使用场景：
# 1. 函数不是你的 —— 比如来自第三方库，没法在上面加 @tool：
#      from some_library import calculate
#      calc_tool = StructuredTool.from_function(
#          func=calculate, name="calculator", description="高级计算器"
#      )
#
# 2. 想给工具起不同的名字或描述 —— 比如函数名是 send_email，
#    但想让 LLM 看到的工具名叫 email_sender，或覆盖 docstring 给 LLM 更清晰的描述。
#
# 3. 动态创建工具 —— 比如在循环中批量注册：
#      tools = []
#      for func in my_functions:
#          tools.append(StructuredTool.from_function(func=func, ...))
#
# 结论：自己写函数且不需要自定义名字/描述 → 用 @tool 更简洁；
#       需要更多控制权或处理已有函数 → 用 StructuredTool.from_function()。
# ====================================================================

def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（模拟）"""
    return f"已发送邮件至 {to}\n主题：{subject}\n内容预览：{body[:50]}..."

email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="发送电子邮件。参数：to(收件人), subject(主题), body(正文)"
)

if __name__ == "__main__":
    print(get_weather.invoke({"city": "北京"}))
    print(query_order.invoke({"order_id": "ORD-2024-0001", "include_details": True}))
    print(email_tool.invoke({
        "to": "user@example.com",
        "subject": "会议通知",
        "body": "请于明天下午2点参加项目评审会议..."
    }))