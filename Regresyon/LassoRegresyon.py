import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso, LassoCV
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
tahmin=ridge_model.predict(X_test)

print("Ridge Train: ",ridge_model.score(X_train,y_train))
print("Ridge Test",ridge_model.score(X_test,y_test))
print("Ridge Test (Diğer Yol): ",mt.r2_score(y_test,tahmin))

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
print("Lasso Train: ",lasso_model.score(X_train,y_train))
print("Lasso Test: ",lasso_model.score(X_test,y_test))

print(ridge_model.coef_)
print(lasso_model.coef_)

print(np.sum(ridge_model.coef_!=0))
print(np.sum(lasso_model.coef_!=0)) #lasso değişkeni modelden dışlayabilir

lamb=LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
print(lamb)