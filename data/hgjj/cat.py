import pandas as pd
import glob

xlsx_files = glob.glob(r"data/hgjj/原始数据excel文件/*.xlsx")

df = pd.concat(
    (pd.read_excel(f) for f in xlsx_files),
    ignore_index=True
)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df['month'] = df['trade_date'].dt.month

# 1. 该月份最后一天
month_end = df['trade_date'] + pd.offsets.MonthEnd(0)
# 2. 取一年中的周序号（ISO 周，1–52/53）
df['week'] = month_end.dt.isocalendar().week

df["category_id"], uniques = pd.factorize(df["category"])
#打印category数量
print(uniques)

df.to_csv("data/hgjj/hgjj.csv", index=False, encoding="utf-8-sig")
