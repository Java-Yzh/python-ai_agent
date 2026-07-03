# 文件读写
# `"w"` = 覆盖写入，`"a"` = 末尾追加，`"r"` = 只读。`with open(...) as f:` 会自动关闭文件，不怕忘。
name = input("请输入您的姓名：")
with open("档案.txt", "w") as f:
    f.write("姓名：" + name + "\n")

age = input("请输入您的年龄：")
with open("档案.txt", "a") as f:
    f.write("年龄：" + age + "\n")

with open("档案.txt", "r") as f:
    print("档案信息：")
    for line in f:
        print("->", line.strip())