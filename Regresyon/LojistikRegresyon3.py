import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

data=pd.read_csv("C:/Users/kerem/Desktop/Programlama/Python/Makine Öğrenmesi/wine.csv")
veri=data.copy()
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

kateori=["3","4","5","6","7","8"]

oe=OrdinalEncoder(categories=[kateori])
veri["Kalite"]=oe.fit_transform(veri["quality"].values.reshape(-1,1))
#print(veri)
#print(veri[veri["quality"]==5][["quality","Kalite"]])

veri=veri.drop(columns="quality",axis=1)

y=veri["Kalite"]
X=veri.drop(columns="Kalite",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

cm=confusion_matrix(y_test,tahmin)
#print(cm)

acs=accuracy_score(y_test,tahmin)
#print(acs*100)

