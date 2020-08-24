# TODO part 1. 머신 러닝에 필요한 모듈을 가져온다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# TODO part 2. 폴더 안에 있는 boston.csv를 가져온다
boston_data = pd.read_csv('boston_csv.csv')

# TODO part 3. boston.csv안에 없는 값들은 열의 평균으로 대체한다.
# na를 어떻게 처리할지가 문제 NaN으로 바꾸면 코드 정상적으로 작동
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(boston_data)
boston_data = imputer.transform(boston_data)

# TODO part 4. boston.csv에 값들에 대해서 요약 통계을 구성한다.
boston_data = pd.DataFrame(boston_data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                                                 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'CAT. MEDV'])
print(boston_data.describe())
print('\n')
corr = boston_data.corr()
sns.heatmap(data=corr, annot=True)
plt.show()

# TODO part 5. 'heatmap’을 활용하여 관계가 높는 값들을 이용해 데이터 셋을 구성한다.
X = pd.DataFrame(np.c_[boston_data['CAT. MEDV'],
                       boston_data['RM']], columns=['CAT. MEDV', 'RM'])
Y = boston_data['MEDV']


# TODO part 6. 머신 러닝 알고리즘 중에서 선형 회귀를 이용하여 머신 러닝을 구현한다.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5)

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

print("The model performance for training set")
print("--------------------------------------")
print('prediction is {}'.format(y_train_predict))
print('RMSE is {}'.format(rmse))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

print("The model performance for testing set")
print("--------------------------------------")
print('prediction is {}'.format(y_test_predict))
print('RMSE is {}'.format(rmse))
