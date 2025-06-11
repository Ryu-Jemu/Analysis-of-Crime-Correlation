import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 한글 폰트 설정 (Mac/Windows 호환)
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
plt.rcParams['axes.unicode_minus'] = False

# --- 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# --- 독립 변수(X)와 종속 변수(y) 정의
X = df[["가로등수", "CCTV수", "인구수"]]
y = df["총범죄"]

# --- 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X, y)

# --- 예측 및 평가
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# --- 회귀 계수 출력
print("회귀계수:")
for feature, coef in zip(X.columns, model.coef_):
    print(f" - {feature}: {coef:.6f}")
print(f"\nR² (설명력): {r2:.4f}")
print(f"평균 제곱 오차 (MSE): {mse:,.2f}")

# --- 예측 vs 실제 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("다중 선형회귀: 실제 vs 예측 총범죄")
plt.xlabel("실제 총범죄")
plt.ylabel("예측 총범죄")
plt.tight_layout()
plt.savefig("multiple_regression_result.png")
plt.show()