# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: clarissa k
RegisterNumber:  212224230047
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
DATA HEAD
<img width="1100" height="181" alt="318743788-2a04d1f8-db54-4611-a509-46d77461e25e" src="https://github.com/user-attachments/assets/044d94fe-e36d-4bcd-867f-357ea0ad06cb" />

<img width="1100" height="198" alt="318744950-db4c1397-3fb3-442b-b4d3-8d51da62aa07" src="https://github.com/user-attachments/assets/b0d41a47-1b17-41b5-9ccb-9d253b3a02dd" />

ISNULL() AND SUM()
<img width="1090" height="76" alt="318747821-12c831d0-592e-4eaa-8015-99bd6f7b625f" src="https://github.com/user-attachments/assets/87ea9d9c-4267-48c0-a432-9dc420c2476b" />

DATA HEAD FOR SALARY
<img width="1100" height="187" alt="318745663-7f5cc54c-0c4d-4ba5-9b38-65ade5e28a01" src="https://github.com/user-attachments/assets/cf0da2be-0d8d-4e86-8982-c4e10f1df163" />

mean squared error:

<img width="1100" height="40" alt="318745802-98f67052-b838-4313-bffa-6669d988205e" src="https://github.com/user-attachments/assets/0742968a-4c2d-4d20-b22b-fb8c64b832d8" />

r2 value:

<img width="1065" height="41" alt="318779545-e6f5cab9-dab9-4c69-bb0e-6fa0abee1da0" src="https://github.com/user-attachments/assets/b7534833-b7eb-473a-a42b-b619d927dddd" />

data prediction:
 
<img width="311" height="38" alt="318779650-92b5c1d6-e495-4eaa-9a9a-8eb3a37ae0bc" src="https://github.com/user-attachments/assets/1b24ce72-d2f5-4ee6-b038-6bfc793a40a2" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
