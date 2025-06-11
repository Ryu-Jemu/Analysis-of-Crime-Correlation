import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import platform

# --- 한글 폰트 설정 (Mac/Windows 호환)
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# ===============================
# CCTV 기준 t-test
# ===============================
median_cctv = df["CCTV수"].median()
group_cctv_high = df[df["CCTV수"] >= median_cctv]
group_cctv_low = df[df["CCTV수"] < median_cctv]

# --- t-test 수행
t_stat_cctv, p_val_cctv = stats.ttest_ind(group_cctv_high["총범죄"],
                                          group_cctv_low["총범죄"],
                                          equal_var=False)

print("[CCTV 수 기준 t-검정]")
print(f" - t 통계량: {t_stat_cctv:.3f}")
print(f" - p-value: {p_val_cctv:.4f}")
if p_val_cctv < 0.05:
    print(" → CCTV 수에 따라 총범죄 발생에 유의미한 차이가 있습니다.\n")
else:
    print(" → CCTV 수에 따라 총범죄 발생에 유의미한 차이가 없습니다.\n")

# --- 박스플롯 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x=["많음"] * len(group_cctv_high) + ["적음"] * len(group_cctv_low),
            y=pd.concat([group_cctv_high["총범죄"], group_cctv_low["총범죄"]]),
            palette="pastel")
plt.title("CCTV 수 기준 총범죄 분포 (t-test)")
plt.xlabel("CCTV 수 그룹")
plt.ylabel("총범죄 건수")
plt.tight_layout()
plt.savefig("t_test_result_cctv.png")
plt.show()

# ===============================
# 가로등 기준 t-test
# ===============================
median_lamp = df["가로등수"].median()
group_lamp_high = df[df["가로등수"] >= median_lamp]
group_lamp_low = df[df["가로등수"] < median_lamp]

# --- t-test 수행
t_stat_lamp, p_val_lamp = stats.ttest_ind(group_lamp_high["총범죄"],
                                          group_lamp_low["총범죄"],
                                          equal_var=False)

print("[가로등 수 기준 t-검정]")
print(f" - t 통계량: {t_stat_lamp:.3f}")
print(f" - p-value: {p_val_lamp:.4f}")
if p_val_lamp < 0.05:
    print(" → 가로등 수에 따라 총범죄 발생에 유의미한 차이가 있습니다.\n")
else:
    print(" → 가로등 수에 따라 총범죄 발생에 유의미한 차이가 없습니다.\n")

# --- 박스플롯 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x=["많음"] * len(group_lamp_high) + ["적음"] * len(group_lamp_low),
            y=pd.concat([group_lamp_high["총범죄"], group_lamp_low["총범죄"]]),
            palette="Set3")
plt.title("가로등 수 기준 총범죄 분포 (t-test)")
plt.xlabel("가로등 수 그룹")
plt.ylabel("총범죄 건수")
plt.tight_layout()
plt.savefig("t_test_result_lamp.png")
plt.show()