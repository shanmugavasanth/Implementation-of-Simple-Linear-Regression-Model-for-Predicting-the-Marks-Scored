# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by : Shanmuga Vasanth M

RegisterNumber : 212223040191

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

### Dataset
![WhatsApp Image 2024-08-28 at 09 00 00_f0dc0467](https://github.com/user-attachments/assets/bf4413d9-6c72-4194-a229-d70ce5031ae3)


### Head Values
![WhatsApp Image 2024-08-28 at 09 00 03_fe795ec3](https://github.com/user-attachments/assets/ab7199a0-bee7-4fc9-87b2-23a99ec90044)


### Tail Values
![WhatsApp Image 2024-08-28 at 08 59 51_b4575e69](https://github.com/user-attachments/assets/35fbd040-4840-4826-ab08-0996ef1d854e)


### X and Y values
![WhatsApp Image 2024-08-28 at 08 59 55_afd9e3c2](https://github.com/user-attachments/assets/6febead6-f89d-470d-8707-40c21823bbcf)


### Predication values of X and Y
![WhatsApp Image 2024-08-28 at 08 59 58_86ad62f6](https://github.com/user-attachments/assets/f5ce083a-9e9a-4ea8-b871-2a842b49b047)


### MSE,MAE and RMSE
![WhatsApp Image 2024-08-28 at 09 00 04_b18804e9](https://github.com/user-attachments/assets/361dfb85-29b5-4090-b060-59ca8fdd4b67)


### Training Set
![WhatsApp Image 2024-08-28 at 08 59 43_598f792c](https://github.com/user-attachments/assets/d3a81c7e-32f3-4ffd-866f-9b17eb6a66bf)


### Testing Set
![WhatsApp Image 2024-08-28 at 09 00 01_05fdaacb](https://github.com/user-attachments/assets/57c61579-7be7-4146-9441-04a53cb83f4b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
