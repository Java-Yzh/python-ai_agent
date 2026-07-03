# 字典（dict）特性：
# 1. 键值对存储：每个元素由 键(key):值(value) 组成，通过键快速取值
# 2. 键唯一性：同一字典中键不能重复，重复赋值会覆盖原有值
# 3. 键的不可变性：键必须是可哈希类型（如 str、int、tuple），不能用 list 或 dict 作键
# 4. 值的任意性：值可以是任意类型，包括列表、字典等，如 {"info": [1, 2, 3]}
# 5. 有序性（Python 3.7+）：字典按插入顺序保留顺序，不再是无序的
# 6. 可变性：创建后可以增、删、改键值对，对象内存地址不变
# 7. 常用取值方式：
#      d[key]       键不存在会报错  | d.get(key, 默认值)  键不存在返回默认值
# 8. 动态增删：
#      d[key] = value   新增或修改  | del d[key]       删除指定键
#      d.pop(key)       删除并返回  | d.clear()        清空字典
#      d.update(other)  批量合并    | d.setdefault(k, v)  不存在才设置
# 9. 遍历方式：
#      d.keys()    所有键    | d.values()   所有值
#      d.items()   所有键值对（for k, v in d.items()）
# 10. 常用判断与统计：
#      "key" in d    判断键是否存在  | len(d)  键值对个数
# 11. 字典推导式：
#      squares = {x: x**2 for x in range(5)}  → {0:0, 1:1, 2:4, 3:9, 4:16}
#
# 总结：字典是 Python 中最常用的映射（关联）数据结构，
#       以键值对存储，查找速度快（接近 O(1)），适合表示"属性-值"关系或配置数据。

# 字典
student = {
    "姓名": "小明",
    "年龄": 20,
    "成绩": 88
}

print(student["姓名"])          # 用钥匙取值
student["成绩"] = 92            # 修改
student["班级"] = "3班"         # 新增

# 遍历所有钥匙和值
for key, value in student.items():
    print(key, ":", value)