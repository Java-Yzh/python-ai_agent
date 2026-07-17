import streamlit as st
import pandas as pd
import time

st.title("缓存实战")

# @st.cache_data：缓存数据（推荐用于 DataFrame、API 调用等）
@st.cache_data(ttl=3600)  # 缓存 1 小时
def load_large_dataset():
    """模拟从数据库或 API 加载大数据集"""
    time.sleep(2)  # 模拟 IO 延迟
    df = pd.DataFrame({
        "id": range(10000),
        "value": range(10000)
    })
    return df

# @st.cache_resource：缓存资源（数据库连接、模型加载等）
@st.cache_resource
def load_ml_model():
    """模拟加载机器学习模型"""
    time.sleep(3)
    return "预训练模型已加载"

# 使用
if st.button("加载数据"):
    df = load_large_dataset()
    st.dataframe(df.head(10))
    st.success("数据已加载（首次慢，后续秒出）")

model = load_ml_model()
st.info(model)