# 循环计算1到100的累加和
total1 = 0
for i in range(1, 101):
    total1 += i
print(total1)


total2 = 0
i = 1
while i <= 100:
    total2 += i
    i += 1
print(total2)