import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/MakineOgrenmesi/Regresyon/kanser.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

veri=data.copy()

veri=veri.drop(columns=["id","Unnamed: 32"],axis=1)

veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]

y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

svm=SVC(random_state=0)
svm.fit(X_train,y_train)
tahmin=svm.predict(X_test)

acs=accuracy_score(y_test,tahmin)
print(acs)

parametreler={"C":range(1,20),"kernel":["linear","poly","rbf"]}
grid=GridSearchCV(estimator=svm,param_grid=parametreler,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)