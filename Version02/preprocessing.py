import pandas as pd
from functools import reduce

# === 경로 설정 ===
DATA_DIR = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/"

# === 1. CCTV 수
df_cctv = pd.read_csv(DATA_DIR + "cctv.csv")
df_cctv['자치구'] = df_cctv['자치구'].str.strip()
cctv_counts = df_cctv['자치구'].value_counts().reset_index()
cctv_counts.columns = ['자치구', 'CCTV수']

# === 2. 범죄 수
df_crime = pd.read_csv(DATA_DIR + "crime.csv", skiprows=3)
df_crime = df_crime.rename(columns={'자치구별(2)': '자치구'})
crime_cols = ['발생', '발생.1', '발생.2', '발생.3', '발생.4', '발생.5']
df_crime[crime_cols] = df_crime[crime_cols].apply(pd.to_numeric, errors='coerce')
df_crime['총범죄'] = df_crime[crime_cols].sum(axis=1)
crime_data = df_crime[['자치구', '총범죄']]

# === 3. 인구 수
df_population = pd.read_csv(DATA_DIR + "population.csv", header=None, skiprows=2)
df_population.columns = ['동별(1)', '자치구', '동별(3)', '인구수', '면적', '인구밀도']
df_population = df_population[df_population['자치구'].str.contains('구')]
df_population = df_population[df_population['자치구'] != '소계']
df_population['인구수'] = pd.to_numeric(df_population['인구수'].astype(str).str.replace(",", ""), errors='coerce')
population_data = df_population[['자치구', '인구수']]

# === 4. 평균소득
df_income = pd.read_csv(DATA_DIR + "average-earnings.csv", encoding='cp949')
df_income['자치구'] = df_income['시군구'].str.replace("서울특별시", "").str.strip()
df_income['평균소득'] = pd.to_numeric(df_income['평균소득월액'], errors='coerce')
income_data = df_income[['자치구', '평균소득']]

# === 5. 유흥주점 수
df_ent = pd.read_csv(DATA_DIR + "seoul-entertainment.csv", encoding='cp949')
df_ent['지번주소'] = df_ent['지번주소'].astype(str)
df_ent['자치구'] = df_ent['지번주소'].str.extract(r'서울특별시\s*(\S+구)')
df_ent = df_ent[df_ent['영업상태명'] == '영업/정상']
entertainment_counts = df_ent['자치구'].value_counts().reset_index()
entertainment_counts.columns = ['자치구', '유흥주점수']

# === 6. 병합
dfs = [cctv_counts, crime_data, population_data, income_data, entertainment_counts]
main_df = reduce(lambda left, right: pd.merge(left, right, on='자치구', how='outer'), dfs)

# 서울 자치구 필터링
seoul_gu_list = [
    '강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구',
    '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구',
    '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구'
]
main_df = main_df[main_df['자치구'].isin(seoul_gu_list)]

# 결측치 처리
main_df = main_df.fillna(0)

# 저장
main_df.to_csv(DATA_DIR + "main.csv", index=False, encoding='utf-8-sig')
print("main.csv 생성 완료")