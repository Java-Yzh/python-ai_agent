import os

DATA_FILE = "账本.csv"

def init_file():
    """如果文件不存在，创建并写入表头"""
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write("类型,金额,备注\n")

def add_record():
    """添加一条收支记录"""
    kind = input("类型（收入/支出）：")
    amount = input("金额：")
    note = input("备注：")

    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(f"{kind},{amount},{note}\n")
    print("记录成功！")

def view_records():
    """查看所有记录并统计"""
    if not os.path.exists(DATA_FILE):
        print("还没有任何记录。")
        return

    income_total = 0
    expense_total = 0

    print("\n--- 收支明细 ---")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # 跳过表头
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        kind, amount, note = parts
        print(f"  {kind} | {amount} 元 | {note}")
        if kind == "收入":
            income_total += float(amount)
        elif kind == "支出":
            expense_total += float(amount)

    print(f"\n总收入：{income_total} 元")
    print(f"总支出：{expense_total} 元")
    print(f"结余：{income_total - expense_total} 元")

def main():
    init_file()
    while True:
        print("\n======== 个人收支管家 ========")
        print("1. 添加记录")
        print("2. 查看记录")
        print("3. 退出")
        choice = input("请选择：")

        if choice == "1":
            add_record()
        elif choice == "2":
            view_records()
        elif choice == "3":
            print("再见！")
            break
        else:
            print("输入错误，请重试。")

main()