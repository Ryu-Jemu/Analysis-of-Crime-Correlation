import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 데이터 로드
file_path = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/merged.csv"
df = pd.read_csv(file_path)

# 분석할 독립 변수만 선택
X = df[['CCTV수', '인구수', '평균소득', '유흥주점수', '가로등수']]

# 상수항 추가 (절편)
X = add_constant(X)

# VIF 계산
vif_data = pd.DataFrame()
vif_data["변수명"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 결과 출력
print(vif_data)