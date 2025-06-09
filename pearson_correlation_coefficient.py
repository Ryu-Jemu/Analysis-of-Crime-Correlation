import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# --- 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
plt.rcParams['axes.unicode_minus'] = False

# --- CSV 불러오기
df_cctv = pd.read_csv("cctv.csv")
df_crime = pd.read_csv("crime.csv")
df_population = pd.read_csv("population.csv")

# --- SHP 파일 불러오기
gdf_streetlamp = gpd.read_file("FSN_30_20221130_G_001.shp")
gdf_streetlamp["위도"] = gdf_streetlamp.geometry.y
gdf_streetlamp["경도"] = gdf_streetlamp.geometry.x

# --- 서울시 범위로 필터링
gdf_seoul = gdf_streetlamp[
    (gdf_streetlamp["위도"] >= 37.4) & (gdf_streetlamp["위도"] <= 37.7) &
    (gdf_streetlamp["경도"] >= 126.7) & (gdf_streetlamp["경도"] <= 127.2)
].copy()

# --- 동 → 자치구 매핑 (생략 가능)
dong_to_gu = {
    '압구정동': '강남구', '청담동': '강남구', '삼성동': '강남구', '개포동': '강남구', '역삼동': '강남구', '일원동': '강남구', '대치동': '강남구',
    '암사동': '강동구', '성내동': '강동구', '천호동': '강동구', '둔촌동': '강동구', '강일동': '강동구',
    '미아동': '강북구', '수유동': '강북구', '번동': '강북구', '우이동': '강북구',
    '가양동': '강서구', '염창동': '강서구', '등촌동': '강서구', '화곡동': '강서구', '내발산동': '강서구', '외발산동': '강서구', '공항동': '강서구',
    '신림동': '관악구', '봉천동': '관악구',
    '광장동': '광진구', '구의동': '광진구', '자양동': '광진구', '중곡동': '광진구', '군자동': '광진구', '화양동': '광진구',
    '개봉동': '구로구', '고척동': '구로구', '구로동': '구로구', '신도림동': '구로구', '오류동': '구로구', '가리봉동': '구로구',
    '당산동': '영등포구', '문래동': '영등포구', '영등포동': '영등포구', '양평동': '영등포구', '신길동': '영등포구', '도림동': '영등포구',
    '마포동': '마포구', '망원동': '마포구', '서교동': '마포구', '성산동': '마포구', '상암동': '마포구', '공덕동': '마포구',
    '가락동': '송파구', '문정동': '송파구', '방이동': '송파구', '석촌동': '송파구', '송파동': '송파구', '잠실동': '송파구',
    '신천동': '송파구', '풍납동': '송파구', '거여동': '송파구', '마천동': '송파구', '오금동': '송파구', '장지동': '송파구', '삼전동': '송파구',
    '노량진동': '동작구', '상도동': '동작구', '신대방동': '동작구', '흑석동': '동작구',
    '이태원동': '용산구', '후암동': '용산구', '동자동': '용산구', '청파동': '용산구', '보광동': '용산구',
    '종로1가': '종로구', '종로2가': '종로구', '종로3가': '종로구', '종로4가': '종로구', '혜화동': '종로구',
    '명륜1가': '종로구', '명륜2가': '종로구', '청운동': '종로구', '창신동': '종로구',
    '을지로1가': '중구', '을지로2가': '중구', '을지로3가': '중구', '을지로4가': '중구', '신당동': '중구', '황학동': '중구',
    '장충동1가': '중구', '장충동2가': '중구', '필동': '중구', '충무로1가': '중구', '충무로2가': '중구',
    '광희동1가': '중구', '광희동2가': '중구',
    '면목동': '중랑구', '묵동': '중랑구', '상봉동': '중랑구', '중화동': '중랑구', '망우동': '중랑구'
}

# 자치구 정리
gdf_seoul["자치구"] = gdf_seoul["LEGALDON_N"].map(dong_to_gu)
gdf_seoul = gdf_seoul.dropna(subset=["자치구"])

# --- 자치구별 집계
streetlamp_count = gdf_seoul.groupby("자치구").size().reset_index(name="가로등수")
cctv_count = df_cctv.groupby("자치구").size().reset_index(name="CCTV수")

# --- 범죄 데이터 가공
df_crime_filtered = df_crime[["자치구별(2)", "2023", "2023.1", "2023.2", "2023.3", "2023.4", "2023.5"]].copy()
df_crime_filtered.columns = ["자치구", "살인", "강도", "성폭력", "절도", "폭력", "기타"]
for col in ["살인", "강도", "성폭력", "절도", "폭력"]:
    df_crime_filtered[col] = pd.to_numeric(df_crime_filtered[col], errors='coerce')
df_crime_filtered["총범죄"] = df_crime_filtered[["살인", "강도", "성폭력", "절도", "폭력"]].sum(axis=1)

# --- 인구 데이터 정리
df_population = df_population[["동별(2)", "2024"]].copy()
df_population.columns = ["자치구", "인구수"]
df_population_grouped = df_population.groupby("자치구", as_index=False).sum()

# --- 병합 순서: 가로등 → CCTV → 범죄 → 인구
df_merge = pd.merge(streetlamp_count, cctv_count, on="자치구", how="outer")
df_merge = pd.merge(df_merge, df_crime_filtered[["자치구", "총범죄"]], on="자치구", how="outer")
df_merge = pd.merge(df_merge, df_population_grouped, on="자치구", how="outer")
df_merge_clean = df_merge.dropna()

# --- 저장
df_merge_clean.to_csv("merged_data.csv", index=False)

# --- 상관계수 분석 및 시각화
correlation_matrix = df_merge_clean[["가로등수", "CCTV수", "총범죄", "인구수"]].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("자치구별 상관관계 분석 (피어슨 상관계수)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()