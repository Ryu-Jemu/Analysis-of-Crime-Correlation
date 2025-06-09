import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import folium
import platform
from folium.plugins import MarkerCluster

# --- 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 병합된 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# --- CCTV 기준 t-test
median_cctv = df["CCTV수"].median()
group_cctv_high = df[df["CCTV수"] >= median_cctv]
group_cctv_low = df[df["CCTV수"] < median_cctv]

t_stat_cctv, p_val_cctv = stats.ttest_ind(group_cctv_high["총범죄"], group_cctv_low["총범죄"], equal_var=False)

# 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x=["많음"] * len(group_cctv_high) + ["적음"] * len(group_cctv_low),
            y=pd.concat([group_cctv_high["총범죄"], group_cctv_low["총범죄"]]),
            palette="pastel")
plt.title("CCTV 수 기준 총범죄 분포 (t-test)")
plt.xlabel("CCTV 수 그룹")
plt.ylabel("총범죄 건수")
plt.tight_layout()
plt.savefig("t_test_result_cctv.png")
plt.close()

# --- 가로등 기준 t-test
median_lamp = df["가로등수"].median()
group_lamp_high = df[df["가로등수"] >= median_lamp]
group_lamp_low = df[df["가로등수"] < median_lamp]

t_stat_lamp, p_val_lamp = stats.ttest_ind(group_lamp_high["총범죄"], group_lamp_low["총범죄"], equal_var=False)

plt.figure(figsize=(8, 6))
sns.boxplot(x=["많음"] * len(group_lamp_high) + ["적음"] * len(group_lamp_low),
            y=pd.concat([group_lamp_high["총범죄"], group_lamp_low["총범죄"]]),
            palette="Set3")
plt.title("가로등 수 기준 총범죄 분포 (t-test)")
plt.xlabel("가로등 수 그룹")
plt.ylabel("총범죄 건수")
plt.tight_layout()
plt.savefig("t_test_result_lamp.png")
plt.close()

# --- 지도 시각화
district_coords = {
    '강남구': (37.5172, 127.0473), '강동구': (37.5301, 127.1238), '강북구': (37.6396, 127.0256),
    '강서구': (37.5509, 126.8495), '관악구': (37.4781, 126.9516), '광진구': (37.5384, 127.0823),
    '구로구': (37.4955, 126.8877), '금천구': (37.4603, 126.9004), '노원구': (37.6542, 127.0568),
    '도봉구': (37.6691, 127.0324), '동대문구': (37.5744, 127.0396), '동작구': (37.5124, 126.9392),
    '마포구': (37.5636, 126.9084), '서대문구': (37.5791, 126.9368), '서초구': (37.4836, 127.0327),
    '성동구': (37.5633, 127.0368), '성북구': (37.5894, 127.0167), '송파구': (37.5145, 127.1065),
    '양천구': (37.5169, 126.8664), '영등포구': (37.5264, 126.8962), '용산구': (37.5324, 126.9901),
    '은평구': (37.6025, 126.9291), '종로구': (37.5730, 126.9794), '중구': (37.5639, 126.9975), '중랑구': (37.6063, 127.0928)
}

df["CCTV_그룹"] = df["CCTV수"].apply(lambda x: "많음" if x >= median_cctv else "적음")

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    gu = row["자치구"]
    coord = district_coords.get(gu)
    if coord:
        folium.Marker(
            location=coord,
            popup=f"{gu} - CCTV: {row['CCTV_그룹']}",
            icon=folium.Icon(color="blue" if row["CCTV_그룹"] == "많음" else "lightgray")
        ).add_to(marker_cluster)

m.save("cctv_group_map.html")