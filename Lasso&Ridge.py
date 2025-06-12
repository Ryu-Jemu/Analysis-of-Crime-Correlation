import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# === 1. 파일 경로 및 데이터 로드 ===
file_path = "/Users/ryujemu/Desktop/Analysis-of-Crime-Correlation/Version02/data/merged.csv"
df = pd.read_csv(file_path)

# === 2. 독립 변수(X)와 종속 변수(y) 설정 ===
X = df[['CCTV수', '인구수', '평균소득', '유흥주점수', '가로등수']]
y = df['총범죄']

# === 3. 결측치 제거 ===
X = X.dropna()
y = y.loc[X.index]

# === 4. 정규화 (표준화) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Lasso 회귀 분석 ===
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
lasso_preds = lasso.predict(X_scaled)

print("📌 Lasso 회귀 결과")
print("선택된 알파(λ):", lasso.alpha_)
print("R² Score:", r2_score(y, lasso_preds))
print("계수:")
for feature, coef in zip(X.columns, lasso.coef_):
    print(f"{feature}: {coef:.4f}")

print("\n" + "="*60 + "\n")

# === 6. Ridge 회귀 분석 ===
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge.fit(X_scaled, y)
ridge_preds = ridge.predict(X_scaled)

print("📌 Ridge 회귀 결과")
print("선택된 알파(λ):", ridge.alpha_)
print("R² Score:", r2_score(y, ridge_preds))
print("계수:")
for feature, coef in zip(X.columns, ridge.coef_):
    print(f"{feature}: {coef:.4f}")