import streamlit as st
import time

st.title("进度与状态反馈")

# 消息提示
st.info("这是一条信息提示")
st.success("操作成功！")
st.warning("请注意，磁盘空间不足")
st.error("发生错误，请重试")

# 进度条
st.subheader("文件处理进度")
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(101):
    progress_bar.progress(i)
    status_text.text(f"处理中…… {i}%")
    time.sleep(0.02)

progress_bar.empty()
status_text.success("处理完成！")

# 加载动画
with st.spinner("正在加载模型……"):
    time.sleep(2)
st.success("模型加载完毕")

# 状态标识
with st.status("正在执行数据清洗……", expanded=True) as status:
    st.write("步骤 1：移除重复值……")
    time.sleep(1)
    st.write("步骤 2：填充缺失值……")
    time.sleep(1)
    st.write("步骤 3：格式化日期列……")
    time.sleep(1)
    status.update(label="清洗完成！", state="complete")