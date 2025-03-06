
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.
2. Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure.
3. Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
4. Train Linear Regression Model: Initialize and train a Linear Regression model using the training data.
5. Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.

## Program and output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOKUL S
RegisterNumber:  212223040051
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```
![ex2_op1](https://github.com/user-attachments/assets/e762caab-7eb8-4caf-98f4-0db0c8326843)
```
x=df.iloc[:,:-1].values
x
```
![ex2_op2](https://github.com/user-attachments/assets/34851270-c92e-4ce8-85f7-793894450e92)

```
y=df.iloc[:,1].values
y
```
![ex2_op3](https://github.com/user-attachments/assets/50ab746d-de2a-409c-ad4e-ee8ee7386a5a)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
```
```
y_pred
```
![ex2_op4](https://github.com/user-attachments/assets/59465be7-b7fd-4822-ba83-e68b94b7b4f6)
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
![ex2_op5](https://github.com/user-attachments/assets/5a643ebc-eabd-412c-b061-83d5d804ae1c)
```
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
![ex2_op6](https://github.com/user-attachments/assets/66297300-f916-4d31-bf55-3ea644b675d0)
```
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title("Hours vs scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
![ex2_op7](https://github.com/user-attachments/assets/e00932ad-f9f4-4e10-975b-3498a7d927c4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
