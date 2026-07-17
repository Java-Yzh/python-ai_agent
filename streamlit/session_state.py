import streamlit as st

st.title("Session State 实战")

# 初始化
if "count" not in st.session_state:
    st.session_state.count = 0
if "todo_items" not in st.session_state:
    st.session_state.todo_items = []

# 计数器
st.subheader("计数器")
col1, col2, col3 = st.columns([1, 1, 2])
if col1.button("➕"):
    st.session_state.count += 1
if col2.button("➖"):
    st.session_state.count -= 1
col3.metric("当前计数", st.session_state.count)

# 待办清单
st.subheader("待办清单")
new_item = st.text_input("添加待办事项")
if st.button("添加") and new_item:
    st.session_state.todo_items.append(new_item)

for i, item in enumerate(st.session_state.todo_items):
    col1, col2 = st.columns([4, 1])
    col1.write(f"{i+1}. {item}")
    if col2.button("删除", key=f"del_{i}"):
        st.session_state.todo_items.pop(i)
        st.rerun()