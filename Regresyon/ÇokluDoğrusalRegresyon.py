import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/advertising.csv")
veri=data.copy()
#print(veri)

#print(veri.corr()["Sales"])

"""sns.pairplot(veri,kind="reg")
plt.show()"""

Q1=veri["Newspaper"].quantile(0.25)
Q3=veri["Newspaper"].quantile(0.75)
IQR=Q3-Q1
ustsınır=Q3+1.5*IQR
aykırı=veri["Newspaper"]>ustsınır
veri.loc[aykırı,"Newspaper"]=ustsınır

"""sns.boxplot(veri["Newspaper"])
plt.show()"""

y=veri["Sales"]
X=veri[["TV","Radio"]]

sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit()
#print(model.summary())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#print(X_train)

lr=LinearRegression()
lr.fit(X_train,y_train)
#print(lr.coef_)

tahmin=lr.predict(X_test)
#print(tahmin)

y_test=y_test.sort_index()

df=pd.DataFrame({"Gerçek":y_test,"Tahmin":tahmin})
df.plot(kind="line")
plt.show()