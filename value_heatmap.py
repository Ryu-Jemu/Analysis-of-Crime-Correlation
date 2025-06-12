import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# CSV 파일 경로
file_path = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/merged.csv"

# windows의 경우
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# mac의 경우
matplotlib.rcParams['font.family'] = 'AppleGothic'

# 데이터 불러오기
df = pd.read_csv(file_path)

# 분석에 필요한 변수만 추출
columns_of_interest = ['CCTV수', '총범죄', '인구수', '평균소득', '유흥주점수', '가로등수']
df_selected = df[columns_of_interest]

# 상관계수 계산
corr_matrix = df_selected.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('주요 변수 간 상관관계 히트맵')
plt.tight_layout()
plt.show()