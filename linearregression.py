import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('./data.csv')
#print(df.head())
x= df.iloc[:,:-1]
y= df.iloc[:,1:]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)
regresion=LinearRegression()
regresion.fit(x_train,y_train)
x1= [[5]]
y_pred=regresion.predict(x1)
print(y_pred)
# plt.scatter(x_train,y_train,color='red')
# plt.plot(x_train,regresion.predict(x_train),color='Blue')
# plt.show()


