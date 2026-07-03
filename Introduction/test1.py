length = int(input("请输入您的身高数据（厘米）："))
height = int(input("请输入您的体重数据（公斤）："))

BMI = height / (length / 100) ** 2
print("您的BMI值为：", BMI)