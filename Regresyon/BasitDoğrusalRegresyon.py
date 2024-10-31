import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/Salary_Data.csv")
veri=data.copy()

#print(veri.isnull().sum())

y=veri["Salary"]
X=veri["YearsExperience"]

"""plt.scatter(X,y)
plt.show()"""

sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit()
#print(model.summary())

lr=LinearRegression()
lr.fit(X.values.reshape(-1,1),y.values.reshape(-1,1))
print(lr.intercept_,lr.coef_)
print(lr.predict(X.values.reshape(-1,1)))