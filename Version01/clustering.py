import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# 정규화할 컬럼
features = ["가로등수", "CCTV수", "총범죄", "인구수"]
X = df[features]

# --- 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMeans 군집 (k=3 사용)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df["클러스터"] = kmeans.fit_predict(X_scaled)

# --- 클러스터별 평균 보기
cluster_summary = df.groupby("클러스터")[features].mean().round(2)
print("\n[클러스터별 평균 지표]")
print(cluster_summary)

# --- 클러스터 시각화 (Pairplot)
sns.pairplot(df, hue="클러스터", vars=features, palette="Set1")
plt.suptitle("군집별 분포", y=1.02)
plt.tight_layout()
plt.savefig("cluster_pairplot.png")
plt.show()

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

color_dict = {0: "red", 1: "blue", 2: "green"}

map_clusters = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
marker_cluster = MarkerCluster().add_to(map_clusters)

for _, row in df.iterrows():
    coord = district_coords.get(row["자치구"])
    if coord:
        folium.Marker(
            location=coord,
            popup=f"{row['자치구']} (Cluster {row['클러스터']})",
            icon=folium.Icon(color=color_dict[row["클러스터"]])
        ).add_to(marker_cluster)

map_clusters.save("cluster_map.html")