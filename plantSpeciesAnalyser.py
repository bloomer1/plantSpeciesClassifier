import numpy as np 
import pandas as pd 
from pandas import Series,DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import linear_model
from sklearn.datasets import load_iris
from scipy import stats

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
m = np.mean(iris_data['Sepal Length'])
print m
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

print pl_data

#plt.hist(data,bins=np.range(min(data),max(data) + binwidth,binwidth))
plt.hist(pl_data,bins=np.arange(1.0,6.9 + 0.59,0.59))
plt.show()

# this is binning by equal width
#plt.hist(pl_data, bins=[0, 10, 20, 30, 40, 50, 100])

sorted_pl_data = sorted(pl_data)
#plt.hist(pl_data, bins=[1, 2, 3, 4, 5, 6, 7])
#plt.show()


#using smoothing by bin means applied on equal depth partitioning
#bin_mean = stats.binned_statistic(pl_data, np.arange(1), statistic='mean', bins=7)
#print bin_mean

# zscore normalization for petal length
# standard deviation taken from description of the data
def zScoreNorm(num):

	return ((num - m)/1.76)

iris_data_c = iris_data
iris_data_c['Petal Length'] = iris_data_c['Petal Length'].apply(zScoreNorm)
norm_zscore_data = iris_data_c['Petal Length']
print "norm_data"
print norm_zscore_data

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





