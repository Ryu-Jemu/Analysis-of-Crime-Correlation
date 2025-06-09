import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')  # Mac
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기 (이미 병합된 csv 사용)
df = pd.read_csv("merged_data.csv")  # 병합된 자치구별 가로등, CCTV, 인구수, 총범죄 데이터

# X, y 정의
X = df[["가로등수", "CCTV수", "인구수"]]
y = df["총범죄"]

# 상수항 추가
X = sm.add_constant(X)

# 회귀 모델 적합
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())

# 예측값 추가
df["예측총범죄"] = model.predict(X)

# 산점도 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x="총범죄", y="예측총범죄", data=df, s=100, edgecolor="w")
plt.plot([df["총범죄"].min(), df["총범죄"].max()],
         [df["총범죄"].min(), df["총범죄"].max()], '--', color='red')
plt.xlabel("실제 총범죄")
plt.ylabel("예측 총범죄")
plt.title("선형회귀: 실제 vs 예측 총범죄")
plt.tight_layout()
plt.savefig("regression_result_fixed.png")
plt.show()