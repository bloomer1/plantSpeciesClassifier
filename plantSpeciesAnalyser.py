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

# Central tendency and dispersion calculation
print 'Mean'
print np.mean(iris_data['Sepal Length'])
#print iris_data['Sepal Length'].mean()

print 'Median'
print np.median(iris_data['Sepal Length'])
#print iris_data['Sepal Length'].median()

print 'Mode'
#print np.mode(iris_data['Sepal Length'])
print iris_data['Sepal Length'].mode()

print 'Quartiles'
print '1st Quartile'
print iris_data['Sepal Length'].quantile(0.25)
print '3rd Quartile'
print iris_data['Sepal Length'].quantile(0.75)
print '4th Quartile'
print iris_data['Sepal Length'].quantile(1.0)

#attribute petal length
#applyting binning for removing noisy data


# this is equal width binning
pl_data = iris_data['Petal Length']

#plt.hist(data,bins=np.range(min(data),max(data) + binwidth,binwidth))
plt.hist(pl_data,bins=np.arange(1.0,6.9 + 0.59,0.59))
plt.show()

#plt.hist(pl_data, bins=[0, 10, 20, 30, 40, 50, 100])





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





