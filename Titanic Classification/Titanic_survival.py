import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data=pd.read_csv('Titanic_Dataset/train.csv')

titanic_data=titanic_data.drop(columns='Cabin', axis=1)

titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)

print(titanic_data['Embarked'].mode()[0])

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)

sns.set()

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

X= titanic_data.drop(columns=['Name','Ticket','Survived'],axis=1)
Y=titanic_data['Survived']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


model=LogisticRegression()
model.fit(X_train,Y_train)

X_train_Prediction=model.predict(X_train)

training_data_accuracy=accuracy_score(Y_train,X_train_Prediction)
print('Accuracy score of training data:',training_data_accuracy)

X_test_Prediction=model.predict(X_test)

test_data_accuracy=accuracy_score(Y_test,X_test_Prediction)
print('Accuracy score of test data:',test_data_accuracy)