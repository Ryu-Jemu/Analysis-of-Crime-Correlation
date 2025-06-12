import pandas as pd
import folium
import statsmodels.api as sm
import os

# === 사용자 환경에 맞게 경로 설정 ===
CSV_PATH = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/merged.csv"
GEOJSON_PATH = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/seoul.geojson"
OUTPUT_DIR = "./output"
OUTPUT_MAP_PATH = os.path.join(OUTPUT_DIR, "crime_residual_map.html")

# 디렉토리 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 데이터 불러오기 ===
df = pd.read_csv(CSV_PATH)

# === 회귀 분석 ===
X = df[['CCTV수', '인구수', '평균소득', '유흥주점수', '가로등수']]
X = sm.add_constant(X)
model = sm.OLS(df['총범죄'], X).fit()
df['예측범죄'] = model.predict(X)
df['잔차'] = df['총범죄'] - df['예측범죄']

# === 지도 시각화 ===
seoul_center = [37.5665, 126.9780]
m = folium.Map(location=seoul_center, zoom_start=11)

# Choropleth: 잔차 시각화
folium.Choropleth(
    geo_data=GEOJSON_PATH,
    data=df,
    columns=["자치구", "잔차"],
    key_on="feature.properties.SIG_KOR_NM",
    fill_color="RdBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="예측 대비 실제 범죄 차이 (잔차)"
).add_to(m)

# 결과 저장
m.save(OUTPUT_MAP_PATH)
print(f"✔️ 시각화 완료! 저장 경로: {OUTPUT_MAP_PATH}")