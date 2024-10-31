import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

veri=pd.read_excel("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/hata.xlsx")
y=veri["Y"]
X=veri[["X1","X2"]]
sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit()
#print(model.summary())

tahmin=model.predict(sabit)
#print(tahmin)

mse=mean_squared_error(y,tahmin)
#print(mse)

rmse=mean_squared_error(y,tahmin,squared=False)
print(rmse)
