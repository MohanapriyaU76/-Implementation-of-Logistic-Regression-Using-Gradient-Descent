# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.

2.Read the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Mohanapriya U 
RegisterNumber:212220040091 
*/
```
import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
        plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)




## Output:
![logistic regression using gradient descent](sam.png)

1.Array value of x:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/0e0f57fb-d043-4589-9a45-4edb3f9bcba7)

2.Array value of y:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/08ee5f41-c1ab-4e22-b7de-146283b4caa5)

3.Exam 1 & 2 score graph:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/0c4f300e-eb38-4879-ae51-d896fcf438f0)

4.Sigmoid graph:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/0ddca39b-e86e-4d08-b34f-98f2a7583eec)

5.J and grad value arry[0,0,0]:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/6033500f-640c-45f2-b769-9ba27b686bd0)

6.J and grad value with array[-24,0.2,0.2]:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/0d5945d0-b9dd-4cef-8d63-df4176a0e46b)

7.res.function & res.x value:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/e1699df5-1dbc-4f35-932d-d52dfedf597d)

8.Decision Boundary graph:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/43e516fb-343b-44eb-9794-a1d6d111962e)

9.Probability value:

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/349da2a6-042b-4b94-9a6b-0479828922b5)

10.Mean prediction value

![image](https://github.com/MohanapriyaU76/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/133958624/743305d5-fc92-437e-bf0d-ffbbf3cec5a8)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

