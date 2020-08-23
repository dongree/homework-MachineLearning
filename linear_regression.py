# TODO part 1. 머신 러닝에 필요한 모듈을 가져온다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# TODO part 2. 폴더 안에 있는 boston.csv를 가져온다
boston_data = pd.read_csv('boston_csv.csv')
print(boston_data.columns)

# TODO part 3. boston.csv안에 없는 값들은 열의 평균으로 대체한다.
# na를 어떻게 처리할지가 문제 NaN으로 바꾸면 코드 정상적으로 작동
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(boston_data)
boston_data = imputer.transform(boston_data)

# TODO part 4. boston.csv에 값들에 대해서 요약 통계을 구성한다.
df = pd.DataFrame(boston_data)
print(df.describe())
sns.heatmap(df)
plt.show()

# TODO part 5. 'heatmap’을 활용하여 관계가 높는 값들을 이용해 데이터 셋을 구성한다.


# TODO part 6. 머신 러닝 알고리즘 중에서 선형 회귀를 이용하여 머신 러닝을 구현한다.
