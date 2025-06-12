import pandas as pd

# === 경로 설정 ===
DATA_DIR = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/"

# === CSV 파일 불러오기 ===
df_main = pd.read_csv(DATA_DIR + "main.csv")
df_estimated = pd.read_csv(DATA_DIR + "estimated-streetlamp.csv")

# === 자치구 기준 병합 ===
df_merged = pd.merge(df_main, df_estimated, on='자치구', how='left')

# === 저장 ===
df_merged.to_csv(DATA_DIR + "merged.csv", index=False, encoding='utf-8-sig')

print("병합 완료")