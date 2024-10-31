import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/maas.csv")
veri=data.copy()

y=veri["Salary"]
X=veri["Level"]
#print(y)

y=np.array(y).reshape(-1,1)
#print(y)
X=np.array(X).reshape(-1,1)

scx=StandardScaler()
scy=StandardScaler()

y=scy.fit_transform(y)
X=scx.fit_transform(X)

svrmodel=SVR(kernel="poly")
svrmodel.fit(X,y)
tahmin=svrmodel.predict(X)

plt.scatter(X,y,c="red")
plt.plot(X,tahmin)
plt.show()