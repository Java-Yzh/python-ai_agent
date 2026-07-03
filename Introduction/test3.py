for i in range(1, 10):
    for j in range(1, i + 1):
        print(j, "*", i, "=", i * j, end="\t")
    # 换行
    print()

print("-----------------------------------------")

# print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
# *objects表示可变参数，即多个参数，参数的个数可以不固定。
# sep表示打印的间隔，默认为空格。
# end表示打印的结尾，默认是换行符。
# file表示输出到哪个文件，默认是sys.stdout。
# flush表示刷新缓冲区，默认是False。
# 循环1：默认 print，每个字符单独一行
for char in "Python":
    print(char)

print("-----------------------------------------")

# 循环2：end=" "，所有字符在同一行，用空格分隔
for char in "Python":
    print(char, end=" ")

print()  # 换行
print("-----------------------------------------")

# 循环3：sep="-"，单个参数时 sep 不生效（sep 只在多个参数之间起作用）
for char in "Python":
    print(char, sep="-", end="")

print()  # 换行
print("-----------------------------------------")

print("A", "B", "C", sep="-", end="")
