import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/advertising.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)

r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)

print("R2: {} MSE: {}".format(r2,mse))

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)
tahmin2=ridge_model.predict(X_test)

r2rid=mt.r2_score(y_test,tahmin2)
mserid=mt.mean_squared_error(y_test,tahmin2)

print("R2 Rid: {} MSE Rid: {}".format(r2rid,mserid))

katsayılar=[]
lambdalar=10**np.linspace(10,2,100)*0.5

for i in lambdalar:
    ridmodel=Ridge(alpha=i)
    ridmodel.fit(X_train,y_train)
    katsayılar.append(ridmodel.coef_)

ax=plt.gca()
ax.plot(lambdalar,katsayılar)
ax.set_xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Katsayılar")
plt.show()