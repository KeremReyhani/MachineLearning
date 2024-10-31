import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/ev.csv")
veri=data.copy()
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"],axis=1,inplace=True)

veri=veri.rename(columns={"X2 house age":"Ev Yaşı",
                 "X3 distance to the nearest MRT station":"Metro Uzaklık",
                 "X4 number of convenience stores":"Market Sayısı",
                 "Y house price of unit area":"Ev Fiyatı"})

"""sns.pairplot(veri)
plt.show()"""

y=veri["Ev Fiyatı"]
X=veri.drop(columns="Ev Fiyatı",axis=1)
#print(X)

pol=PolynomialFeatures(degree=2)
X_pol=pol.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_pol,y,test_size=0.2,random_state=42)

pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train)
tahmin=pol_reg.predict(X_test)

r2=mt.r2_score(y_test,tahmin)
Mse=mt.mean_squared_error(y_test,tahmin)

print("R2: {} MSE: {}".format(r2,Mse))
