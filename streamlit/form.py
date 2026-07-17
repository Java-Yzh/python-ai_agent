import streamlit as st

st.title("用户注册表单")

with st.form("register_form"):
    st.subheader("请填写注册信息")

    username = st.text_input("用户名 *", placeholder="至少 4 个字符")
    email = st.text_input("邮箱 *", placeholder="example@company.com")
    password = st.text_input("密码 *", type="password")
    confirm_password = st.text_input("确认密码 *", type="password")

    col1, col2 = st.columns(2)
    age = col1.number_input("年龄", 18, 100, 25)
    department = col2.selectbox("部门", ["技术部", "产品部", "市场部", "财务部"])

    subscribe = st.checkbox("订阅周报", value=True)

    submitted = st.form_submit_button("注册")

    if submitted:
        if not username or not email or not password:
            st.error("请填写所有必填项（标注 * 的字段）")
        elif password != confirm_password:
            st.error("两次密码输入不一致")
        else:
            st.success(f"注册成功！欢迎 {username}（{department}）")
            st.json({
                "用户名": username,
                "邮箱": email,
                "部门": department,
                "订阅": subscribe
            })

# 对比：不用表单的效果
st.divider()
st.caption("↑ 表单方式（批量提交） vs ↓ 非表单方式（每次改动都触发）")
st.text_input("随便输入点什么看看区别...")