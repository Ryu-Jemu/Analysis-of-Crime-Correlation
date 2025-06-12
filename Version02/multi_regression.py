import pandas as pd
import statsmodels.api as sm

# === 1. 경로 수정 ===
file_path = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/merged.csv"
# === 2. 데이터 불러오기 ===
df = pd.read_csv(file_path)

# === 3. 회귀 분석 대상 변수 설정 ===
X = df[['CCTV수', '인구수', '평균소득', '유흥주점수', '가로등수']]
y = df['총범죄']

# === 4. 결측치 제거 ===
X = X.dropna()
y = y.loc[X.index]

# === 5. 회귀 모델 학습 ===
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# === 6. 결과 출력 ===
print(model.summary())