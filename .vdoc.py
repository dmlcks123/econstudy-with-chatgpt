# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import PolynomialFeatures

- 고정된 랜덤 시드 설정

np.random.seed(42)

- 데이터 생성
X1 = np.linspace(0, 100, 100)  # X1의 범위를 크게 설정

X2 = np.sin(X1) + 0.5 * X1 + np.random.normal(0, 10, 100)  # 큰 노이즈 추가

- 종속 변수 생성 (y = 2*X1 + 0.1*X2^2 - 10*X2 + 100 + 큰 노이즈 추가)
y = 2 * X1 + 0.1 * (X2 ** 2) - 10 * X2 + 100 + np.random.normal(0, 50, 100)

- 데이터프레임으로 정리
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
#
#
#
#
#
#
#
#
#
# 히스토그램 예시
df.hist(bins=30, figsize=(15, 10))

plt.tight_layout()

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 단순선형회귀 모델 (X1만 사용)

model_simple = LinearRegression()

model_simple.fit(df[['X1']], df['y'])

y_pred_simple = model_simple.predict(df[['X1']])

plt.figure(figsize=(6, 4))

plt.scatter(df['X1'], df['y'], color='blue', label='Actual Data')

plt.plot(df['X1'], y_pred_simple, color='red', label='Simple Linear Regression')

plt.title('Simple Linear Regression')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
#
#
- 다중선형회귀 모델

model_multiple = LinearRegression()

model_multiple.fit(df[['X1', 'X2']], df['y'])

y_pred_multiple = model_multiple.predict(df[['X1', 'X2']])

plt.figure(figsize=(6, 4))

plt.scatter(df['X1'], df['y'], color='blue', label='Actual Data')

plt.plot(df['X1'], y_pred_multiple, color='red', label='Multiple Linear Regression')

plt.title('Multiple Linear Regression')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
# 릿지회귀 모델

model_ridge = Ridge(alpha=8.0)

model_ridge.fit(df[['X1', 'X2']], df['y'])

y_pred_ridge = model_ridge.predict(df[['X1', 'X2']])

plt.figure(figsize=(6, 4))

plt.scatter(df['X1'], df['y'], color='blue', label='Actual Data')

plt.plot(df['X1'], y_pred_ridge, color='red', label='Ridge Regression')

plt.title('Ridge Regression')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
#
#
# 라쏘회귀 모델

model_lasso = Lasso(alpha=8)

model_lasso.fit(df[['X1', 'X2']], df['y'])

y_pred_lasso = model_lasso.predict(df[['X1', 'X2']])

plt.figure(figsize=(6, 4))

plt.scatter(df['X1'], df['y'], color='blue', label='Actual Data')

plt.plot(df['X1'], y_pred_lasso, color='red', label='Lasso Regression')

plt.title('Lasso Regression')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
#
# 다항회귀 모델 (2차 다항식)

poly_features = PolynomialFeatures(degree=2)

X_poly_transformed = poly_features.fit_transform(df[['X1', 'X2']])

model_poly = LinearRegression()

model_poly.fit(X_poly_transformed, df['y'])

y_pred_poly = model_poly.predict(X_poly_transformed)

plt.figure(figsize=(6, 4))

plt.scatter(df['X1'], df['y'], color='blue', label='Actual Data')

plt.plot(df['X1'], y_pred_poly, color='red', label='Polynomial Regression')

plt.title('Polynomial Regression')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
# 결과출력 (모델별 회귀 계수)

print("Simple Linear Regression Coefficients:", model_simple.coef_)

print("Multiple Linear Regression Coefficients:", model_multiple.coef_)

print("Ridge Regression Coefficients:", model_ridge.coef_)

print("Lasso Regression Coefficients:", model_lasso.coef_)

print("Polynomial Regression Coefficients:", model_poly.coef_)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 결과 비교 시각화

plt.figure(figsize=(8, 6))

# 단순선형회귀

plt.plot(df['X1'], y_pred_simple, label='Simple Linear Regression', color='red', linewidth=2)

# 다중선형회귀 (선 굵기 증가)

plt.plot(df['X1'], y_pred_multiple, label='Multiple Linear Regression', color='blue', linewidth=4, linestyle='--')

# 릿지회귀 (선 굵기 증가 및 색상 대비 증가)

plt.plot(df['X1'], y_pred_ridge, label='Ridge Regression', color='green', linewidth=4, linestyle=':')

# 라쏘회귀

plt.plot(df['X1'], y_pred_lasso, label='Lasso Regression', color='orange', linewidth=2)

# 다항회귀

plt.plot(df['X1'], y_pred_poly, label='Polynomial Regression', color='purple', linewidth=2)

# 실제 데이터
plt.scatter(df['X1'], df['y'], color='black', label='Actual Data', alpha=0.6)

plt.title('Comparison of Different Regression Models')

plt.xlabel('X1')

plt.ylabel('y')

plt.legend()

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 모델별 예측 값

y_pred_simple = model_simple.predict(df[['X1']])

y_pred_multiple = model_multiple.predict(df[['X1', 'X2']])

y_pred_ridge = model_ridge.predict(df[['X1', 'X2']])

y_pred_lasso = model_lasso.predict(df[['X1', 'X2']])

y_pred_poly = model_poly.predict(X_poly_transformed)

# R² 계산

r2_simple = r2_score(df['y'], y_pred_simple)

r2_multiple = r2_score(df['y'], y_pred_multiple)

r2_ridge = r2_score(df['y'], y_pred_ridge)

r2_lasso = r2_score(df['y'], y_pred_lasso)

r2_poly = r2_score(df['y'], y_pred_poly)

# MAE 계산

mae_simple = mean_absolute_error(df['y'], y_pred_simple)

mae_multiple = mean_absolute_error(df['y'], y_pred_multiple)

mae_ridge = mean_absolute_error(df['y'], y_pred_ridge)

mae_lasso = mean_absolute_error(df['y'], y_pred_lasso)

mae_poly = mean_absolute_error(df['y'], y_pred_poly)

# MSE 계산

mse_simple = mean_squared_error(df['y'], y_pred_simple)

mse_multiple = mean_squared_error(df['y'], y_pred_multiple)

mse_ridge = mean_squared_error(df['y'], y_pred_ridge)

mse_lasso = mean_squared_error(df['y'], y_pred_lasso)

mse_poly = mean_squared_error(df['y'], y_pred_poly)

# RMSE 계산

rmse_simple = np.sqrt(mse_simple)

rmse_multiple = np.sqrt(mse_multiple)

rmse_ridge = np.sqrt(mse_ridge)

rmse_lasso = np.sqrt(mse_lasso)

rmse_poly = np.sqrt(mse_poly)

# 결과 출력

print(f"Simple Linear Regression: R²={r2_simple:.3f}, MAE={mae_simple:.3f}, MSE={mse_simple:.3f}, RMSE={rmse_simple:.3f}")

print(f"Multiple Linear Regression: R²={r2_multiple:.3f}, MAE={mae_multiple:.3f}, MSE={mse_multiple:.3f}, RMSE={rmse_multiple:.3f}")

print(f"Ridge Regression: R²={r2_ridge:.3f}, MAE={mae_ridge:.3f}, MSE={mse_ridge:.3f}, RMSE={rmse_ridge:.3f}")

print(f"Lasso Regression: R²={r2_lasso:.3f}, MAE={mae_lasso:.3f}, MSE={mse_lasso:.3f}, RMSE={rmse_lasso:.3f}")

print(f"Polynomial Regression: R²={r2_poly:.3f}, MAE={mae_poly:.3f}, MSE={mse_poly:.3f}, RMSE={rmse_poly:.3f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 인터넷에 업로드 되어 있는 자료를 불러들이기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# URL of the .dta file
url = "http://www.stata-press.com/data/r9/cancer.dta"
file_name = "cancer.dta"

# Set headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

# Download the file
response = requests.get(url, headers=headers)
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print("File downloaded successfully")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Read the .dta file into a pandas DataFrame
df = pd.read_stata(file_name)
#
#
#
#
#
#
#
#
#
# 종속 변수와 독립 변수 설정
X = df[['studytime', 'drug', 'age']]
y = df['died']

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# 데이터셋 분할 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=45)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. 로지스틱 회귀
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_pred_logistic = model_logistic.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# 2. K-최근접 이웃
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# 3. 서포트 벡터 머신
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
print("Support Vector Machine Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 4. 결정 트리
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# 5. 랜덤 포레스트
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 6. 나이브 베이즈
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
