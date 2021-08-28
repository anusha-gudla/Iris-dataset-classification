import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LogisticRegression


d=pd.read_csv("/content/iris.csv")
d.head()

dummies=pd.get_dummies(d.iloc[:,-1])
dummies

merge=pd.concat([d,dummies],axis='columns')
merge

final=merge.drop(['state'],axis='columns')
final.head()

final=final.drop(['Iris-setosa'],axis='columns')
final.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
newd=d
newd.state=le.fit_transform(newd.state)
newd

d.info()

x=d.iloc[:,:4]
y=d.iloc[:,-1]
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=5)

model=LogisticRegression(random_state=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)#y_test
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

import seaborn as sns
sns.set_style("whitegrid")
sns.FacetGrid(d, hue ="state",
              height = 6).map(plt.scatter,
                              'sepal length',
                              'petal length').add_legend()

