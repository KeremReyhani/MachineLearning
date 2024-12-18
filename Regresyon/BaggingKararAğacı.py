import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as mt
from sklearn.ensemble import BaggingRegressor

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/Regresyon/advertising.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

dtmodel=DecisionTreeRegressor(random_state=0,max_leaf_nodes=18,min_samples_split=4)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)

r2=mt.r2_score(y_test,dttahmin)
rmse=mt.mean_squared_error(y_test,dttahmin,squared=False)

print("Karar Ağacı R2: {} Karar Ağacı RMSE: {}".format(r2,rmse))

bgmodel=BaggingRegressor(random_state=0,n_estimators=23)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)

r22=mt.r2_score(y_test,bgtahmin)
rmse2=mt.mean_squared_error(y_test,bgtahmin,squared=False)

print("Bagging R2: {} Bagging RMSE: {}".format(r22,rmse2))

"""parametreler1={"min_samples_split":range(2,25),"max_leaf_nodes":range(2,25)}
grid1=GridSearchCV(estimator=dtmodel,param_grid=parametreler1)
grid1.fit(X_train,y_train)
print(grid1.best_params_)

parametreler2={"n_estimators":range(2,25)}
grid2=GridSearchCV(estimator=bgmodel,param_grid=parametreler2,cv=10)
grid2.fit(X_train,y_train)
print(grid2.best_params_)"""