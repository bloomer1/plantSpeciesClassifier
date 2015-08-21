import numpy as np 
import pandas as pd 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import linear_model
from sklearn.datasets import load_iris

sns.set_style('whitegrid')


def flower_type(num):
	if num == 0:
		return 'Setosa'
	elif num == 1:
		return 'VersiColour'
	else:
		return 'Virginica'


iris = load_iris()
X = iris.data 
Y = iris.target

print iris.DESCR

iris_data = DataFrame(X,columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

iris_target = DataFrame(Y,columns=['Species'])

print iris_data
print iris_target

iris_target['Species'] = iris_target['Species'].apply(flower_type)
print iris_target.head()
print iris_target.tail()


iris = pd.concat([iris_data,iris_target],axis=1)
print iris 

sns.pairplot(iris,hue='Species',size=2)
sns.plt.show()


sns.factorplot('Petal Length',data=iris,hue='Species',size=8,kind='count')
sns.plt.show()





