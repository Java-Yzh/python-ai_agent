import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title("图表集成")

# 生成模拟数据
df = pd.DataFrame({
    "月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
    "销售额": [120, 135, 160, 150, 180, 210],
    "利润": [30, 38, 45, 42, 50, 62],
    "品类": ["A", "B", "A", "B", "A", "B"]
})

# --- Streamlit 原生图表 ---
st.subheader("原生折线图")
st.line_chart(df.set_index("月份")[["销售额", "利润"]])

st.subheader("原生柱状图")
st.bar_chart(df.set_index("月份")[["销售额"]])

st.subheader("原生面积图")
st.area_chart(df.set_index("月份")[["销售额", "利润"]])

# --- Matplotlib ---
st.subheader("Matplotlib")
fig, ax = plt.subplots()
ax.bar(df["月份"], df["销售额"], color="steelblue")
ax.set_ylabel("销售额（万元）")
ax.set_title("月度销售额")
st.pyplot(fig)

# --- Plotly ---
st.subheader("Plotly 交互图表")
fig2 = px.line(df, x="月份", y="销售额", color="品类",
               markers=True, title="各品类销售趋势")
st.plotly_chart(fig2, use_container_width=True)