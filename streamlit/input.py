import streamlit as st

st.title("输入组件大集合")

# 文本输入
name = st.text_input("你的名字", placeholder="请输入姓名")
password = st.text_input("密码", type="password")
feedback = st.text_area("反馈意见", height=100)

# 数字输入
age = st.number_input("年龄", min_value=0, max_value=150, value=25)
rating = st.slider("满意度评分", 0, 10, 7)

# 选择类
option = st.selectbox("最喜欢的语言", ["Python", "Rust", "Go", "TypeScript"])
colors = st.multiselect("喜欢的颜色", ["红色", "蓝色", "绿色", "黄色"])
agree = st.checkbox("我同意条款")
gender = st.radio("性别", ["男", "女", "其他"])

# 按钮
# 在页面上渲染一个标题为「点击我」的按钮。当用户点击了它，返回 True；没有点击（或页面刚加载时），返回 False
# st.button 和 st.checkbox 都返回布尔值，但区别在于：
# st.checkbox — 状态保持，勾上后一直是 True，取消勾选才变 False
# st.button — 瞬时的，只有点击的那一下是 True，下一次刷新就回到 False
if st.button("点击我"):
    # 仅在按钮被点击时执行，用 st.success 显示一条绿色成功消息，内容引用了上面定义的 name（文本输入）和 option（下拉选择）
    st.success(f"你好 {name}，你选了 {option}！")
    # 放一个庆祝气球动画
    st.balloons()
    st.snow()

# 日期时间
date = st.date_input("选择日期")
time = st.time_input("选择时间")

# 文件上传
uploaded_file = st.file_uploader("上传文件", type=["csv", "txt", "pdf"])
if uploaded_file:
    st.write(f"文件名：{uploaded_file.name}")