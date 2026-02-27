# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas
2. import Decision tree classifier
3. fit the data in the model
4. find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KAVIYA V M
RegisterNumber:212224040154  
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position", "Level"]]
x.head()
y = data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
y_pred
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2 = r2_score(y_test, y_pred)
print(r2)
dt.predict([[5,6]])
```

## Output:

<img width="395" height="429" alt="Screenshot 2026-02-16 092021" src="https://github.com/user-attachments/assets/bbbe2229-c442-4efa-a5c5-f018ca2acd1e" />


<img width="344" height="249" alt="Screenshot 2026-02-16 090137" src="https://github.com/user-attachments/assets/1fe522cb-5ede-47ec-90a9-27ebefab1754" />


<img width="1279" height="107" alt="Screenshot 2026-02-16 092135" src="https://github.com/user-attachments/assets/b3bb7125-66ec-49a8-ae71-fb371de965aa" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
