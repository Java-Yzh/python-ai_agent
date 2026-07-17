import streamlit as st
import pandas as pd
import numpy as np

st.title("数据展示示例")

# 各级标题
st.header("一级标题")
st.subheader("二级标题")
st.caption("这是一行注释文字")

# 代码块
code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language="python")

# 数据表格
df = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["销售额", "利润", "增长率"]
)
st.dataframe(df)          # 交互式表格（可排序/缩放）
st.table(df.head(5))      # 静态表格

# 指标卡片
col1, col2, col3 = st.columns(3)
col1.metric("温度", "24°C", "2°C")
col2.metric("湿度", "65%", "-5%")
col3.metric("风速", "12 km/h", "3 km/h")

# JSON 展示
st.json({"name": "Streamlit", "version": "1.x", "python": "3.12"})