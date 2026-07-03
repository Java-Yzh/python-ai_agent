# 函数
def greet(name):
    """向指定的人打招呼"""
    return "你好，" + name + "！"

def add(a, b):
    return a + b

# 调用
print(greet("小红"))
result = add(3, 5)
print("3 + 5 =", result)

# 有默认参数的函数
def power(x, n=2):    # n 默认是 2
    return x ** n

print(power(5))       # 5 的平方
print(power(5, 3))    # 5 的 3 次方