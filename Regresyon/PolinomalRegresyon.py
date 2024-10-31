import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_excel("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/polinom.xlsx")
veri=data.copy()

y=veri["Verim"]
X=veri["Sıcaklık"]

"""plt.scatter(X,y)
plt.show()"""

y=y.values.reshape(-1,1)
X=X.values.reshape(-1,1)

lr=LinearRegression()
lr.fit(X,y)

tahmin=lr.predict(X)
r2dog=mt.r2_score(y,tahmin)
mse=mt.mean_squared_error(y,tahmin)

print("Doğrusal R2 = {} Doğrusal MSE = {}".format(r2dog,mse))

pol=PolynomialFeatures(degree=2)
X_pol=pol.fit_transform(X)

lr2=LinearRegression()
lr2.fit(X_pol,y)
tahmin2=lr2.predict(X_pol)

r2pol=mt.r2_score(y,tahmin2)
msepol=mt.mean_squared_error(y,tahmin2)

print("Polinomsal R2 = {} Doğrusal MSE = {}".format(r2pol,msepol))

plt.scatter(X,y,c="r")
plt.plot(X,tahmin,c="blue")
plt.plot(X,tahmin2,c="green")
plt.show()