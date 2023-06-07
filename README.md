# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset
2. Check for null and duplicate values
3. Assign x and y values
4. Split the data into training and testing data
5. Import logistic regression and fit the training data
6. Predict y value
7. Calculate accuracy and confusion matrix

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohanapriya U
RegisterNumber:212220040091

import pandas as pd

data=pd.read_csv('/content/Placement_Data.csv')


data.head()

data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)

data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])

data1["status"]=le.fit_transform(data1["status"])

data1

x=data1.iloc[:,:-1]

x

y=data1["status"]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix

confusion=(y_test,y_pred)

confusion

from sklearn.metrics import classification_report

cr=classification_report(y_test,y_pred)

print(cr)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```
## Output:

## PLACEMENT DATA:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/4a8f4795-6f8a-4001-bf67-241dc802fb5c)
## SALARY DATA:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/d5ee33ad-1b02-4482-b4a6-e0be58e25e72)
## CHECKING NULL VALUE:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/762f5d3f-1d86-4df8-b5a7-597c066e956f)
## DATA DUPLICATE:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/544ede4f-ecc0-49d7-915c-24a407930373)
## PRINT DATA:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/39724df8-e293-4047-a1dc-a31467d0cd8b)
## Y_PREDICTED ARRAY:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/68015e7e-f105-4f28-9499-1fe9995c027a)
## CONFUSION ARRAY:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/bfb8c5b2-87e7-4def-b1cd-d2ced2172d3d)
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/25b2ba47-83dc-4c65-bd6c-5c9a8879a6e6)
## CLASSIFICATION REPORT:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/2aa1f569-3280-469e-ab99-e30a582ad14a)
## PREDICTION OF LR:
![image](https://github.com/srimathi-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114581999/a5bde930-23e5-44e7-98e0-511b0440921e)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
