import pandas as pd 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data 
Y = iris.target


#Logistic regresssion
log_reg = LogisticRegression()
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.4, random_state=3)
log_reg.fit(X_train,Y_train)
Y_pred = log_reg.predict(X_test)
print metrics.accuracy_score(Y_test,Y_pred)


#k nearest neighbor

#finding the optimized value for k
k_range = range(1,26)
Y_preds = []
accuracy = []

for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,Y_train)
	Y_pred = knn.predict(X_test)
	Y_preds.append(Y_pred)
	accuracy.append(metrics.accuracy_score(Y_test,Y_pred))


opt_k = 1
max_acc = 0

for idx,acc in enumerate(accuracy):
	if (acc > max_acc):
		max_acc = acc
		opt_k = idx


opt_k = opt_k + 1

print opt_k 

print Y_preds[opt_k - 1]
print max_acc







