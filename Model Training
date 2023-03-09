import pandas as pd
import csv
data=pd.read_csv('/Users/ynosoni/Desktop/513/Rock_Music2.csv')
data
#Drop 'Song Name'
data=data.drop(['Song name'],axis=1)
#Convert the value in the Genre column to a numeric value
genre_values=data['Genre'].unique().tolist()
genre_values
genre_dic={}
for i in range(len(genre_values)):
    genre_dic[genre_values[i]]=i
genre_dic
data['Genre']=data['Genre'].map(genre_dic)
data
#Plotting variable correlation heat map
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
# fig.savefig('corr.png')
plt.show()
#Predicting Genre Using Random Forest Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

taget=data['Genre']
datax=data.drop(['Genre'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(datax,taget,test_size=0.2,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#Using the random forest model
model=RandomForestClassifier(n_estimators=100,max_depth=10)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred

#Model Evaluation
print('Accuracy：',accuracy_score(y_test,y_pred))
print('Confusion Matrix：',confusion_matrix(y_test,y_pred))
print('Classfication Report：',classification_report(y_test,y_pred))
#Using Decision Tree Models to Predict Genre
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred

#Model Evaluation
print('Accuracy：',accuracy_score(y_test,y_pred))
print('Confusion Matrix：',confusion_matrix(y_test,y_pred))

#Predicting Genre using KNN models
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


#Model Evaluation
print('Accuracy：',accuracy_score(y_test,y_pred))
print('Confusion Matrix：',confusion_matrix(y_test,y_pred))

#Predicting Genre using SVM models
from sklearn.svm import SVC
model=SVC(kernel='rbf',C=1.0,gamma='auto')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Model Evaluation
print('Accuracy:：',accuracy_score(y_test,y_pred))
print('Confusion Matrix：',confusion_matrix(y_test,y_pred))

#Using logistic regression models to predict Genre
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Optimize model accuracy
#Adjusting model parameters using grid search
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators':[100,200,300,400,500],
            'max_depth':[5,10,15,20,25,None],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2,4],
            'bootstrap':[True,False]}
grid_search=GridSearchCV(model,param_grid,cv=5)
grid_search.fit(X_train,y_train)
grid_search.best_params_

#Retrain the model using optimal parameters]
# model=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=2,min_samples_leaf=1,bootstrap=True)
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# y_pred
