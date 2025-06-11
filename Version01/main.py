import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import folium

# --- 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# (선택) 자치구 위경도 매핑
gu_location = {
    '강남구': [37.5172, 127.0473], '강동구': [37.5301, 127.1238], '강북구': [37.6398, 127.0255], '강서구': [37.5509, 126.8495],
    '관악구': [37.4781, 126.9516], '광진구': [37.5385, 127.0823], '구로구': [37.4955, 126.8877], '금천구': [37.4569, 126.8958],
    '노원구': [37.6542, 127.0568], '도봉구': [37.6691, 127.0324], '동대문구': [37.5744, 127.0396], '동작구': [37.5124, 126.9396],
    '마포구': [37.5663, 126.9014], '서대문구': [37.5790, 126.9368], '서초구': [37.4837, 127.0324], '성동구': [37.5633, 127.0367],
    '성북구': [37.5894, 127.0167], '송파구': [37.5145, 127.1059], '양천구': [37.5170, 126.8666], '영등포구': [37.5264, 126.8963],
    '용산구': [37.5323, 126.9907], '은평구': [37.6027, 126.9291], '종로구': [37.5729, 126.9793], '중구': [37.5634, 126.9976],
    '중랑구': [37.6063, 127.0927]
}
df["위도"] = df["자치구"].map(lambda x: gu_location.get(x, [None, None])[0])
df["경도"] = df["자치구"].map(lambda x: gu_location.get(x, [None, None])[1])

# ========== 1. 선형 회귀 분석 ==========
X = df[["인구수"]]
y = df["총범죄"]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

r2 = r2_score(y, pred)
mse = mean_squared_error(y, pred)

plt.figure(figsize=(8, 6))
sns.scatterplot(x="인구수", y="총범죄", data=df)
plt.plot(df["인구수"], pred, color="red", label="회귀선")
plt.title(f"선형 회귀: 인구수 → 총범죄 (R²={r2:.2f})")
plt.xlabel("인구수")
plt.ylabel("총범죄")
plt.legend()
plt.tight_layout()
plt.savefig("regression_population.png")
plt.close()

# ========== 2. 군집 분석 ==========
features = df[["가로등수", "CCTV수", "총범죄", "인구수"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["클러스터"] = kmeans.fit_predict(features)

cluster_summary = df.groupby("클러스터").agg({
    "자치구": list,
    "가로등수": "mean",
    "CCTV수": "mean",
    "총범죄": "mean",
    "인구수": "mean"
}).reset_index()

# ========== 3. 클러스터 지도 시각화 ==========
seoul_center = [37.5665, 126.9780]
m = folium.Map(location=seoul_center, zoom_start=11)
colors = ['red', 'blue', 'green']

for i, row in df.iterrows():
    if pd.notnull(row["위도"]) and pd.notnull(row["경도"]):
        popup_text = f"{row['자치구']} (클러스터 {row['클러스터']})<br>총범죄: {row['총범죄']}건"
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=6,
            popup=popup_text,
            color=colors[row["클러스터"]],
            fill=True,
            fill_color=colors[row["클러스터"]],
            fill_opacity=0.7
        ).add_to(m)

m.save("cluster_map.html")

# ========== 결과 저장 및 출력 ==========
df.to_csv("merged_data_with_cluster.csv", index=False)

print("✅ 선형 회귀 결과 (인구수 → 총범죄)")
print(f" - R²: {r2:.4f}")
print(f" - MSE: {mse:.2f}")
print("\n✅ 클러스터 분석 요약:")
print(cluster_summary)