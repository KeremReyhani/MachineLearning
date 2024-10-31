import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso, ElasticNet,ElasticNetCV
import sklearn.metrics as mt

# Boston veri setini yüklemek için URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Veriyi indirip işleme (sep="\\s+" ifadesi ile düzeltildi)
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
df = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Veri ve hedefleri pandas DataFrame olarak birleştirin
data = pd.DataFrame(df, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

# 'Target' sütununu ekleyin
data['PRICE'] = target

# Geniş tablo görüntüleme ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Veri kopyası oluşturma
veri = data.copy()

# Veri çerçevesini ekrana yazdırma

y=veri["PRICE"]
X=veri.drop(columns="PRICE",axis=1)
#print(X)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)

print(ridge_model.score(X_train,y_train))
print(lasso_model.score(X_train,y_train))
print(elas_model.score(X_train,y_train))

print(ridge_model.score(X_test,y_test))
print(lasso_model.score(X_test,y_test))
print(elas_model.score(X_test,y_test))

tahminrid=ridge_model.predict(X_test)
tahminlasso=lasso_model.predict(X_test)
tahminelas=elas_model.predict(X_test)

print(mt.mean_squared_error(y_test,tahminrid))
print(mt.mean_squared_error(y_test,tahminlasso))
print(mt.mean_squared_error(y_test,tahminelas))

lamb=ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_

elas_model2=ElasticNet(alpha=lamb)
elas_model2.fit(X_train,y_train)

print(elas_model2.score(X_train,y_train))
print(elas_model2.score(X_test,y_test))

tahminelas2=elas_model2.predict(X_test)
print(mt.mean_squared_error(y_test,tahminelas2))