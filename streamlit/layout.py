import streamlit as st

st.title("布局管理")

# 侧边栏
with st.sidebar:
    st.header("侧边栏")
    page = st.radio("导航", ["首页", "数据分析", "设置"])
    st.file_uploader("上传数据")

# 多列布局
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.metric("总销售额", "¥1,250,000", "12%")
with col2:
    st.metric("订单数", "3,210", "-8%")
with col3:
    st.metric("客单价", "¥389", "5%")

# 标签页
tab1, tab2, tab3 = st.tabs(["概览", "详情", "导出"])
with tab1:
    st.write("概览内容")
with tab2:
    st.write("详情内容")
with tab3:
    st.write("导出功能")

# 可展开容器
with st.expander("点击展开详细信息"):
    st.write("这里放折叠的内容，适合放不太重要的说明或详细数据。")

# 空占位符（动态更新用）
placeholder = st.empty()
if st.button("更新内容"):
    placeholder.success("内容已动态更新！")

# 容器分组
container = st.container(border=True)
container.write("这是一个带边框的容器")