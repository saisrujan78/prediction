import numpy as np
import pandas as pd
dataset=pd.read_csv('admissionPredict.csv')
dataset
dataset.shape
dataset.columns
x=dataset.iloc[:,1:-1].values 
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
gre_score=reg.predict(x_test)
aucscore=reg.score(x_test,y_test) 
aucscore


