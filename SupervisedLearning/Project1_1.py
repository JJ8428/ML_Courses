'''
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))
'''

# All modules to use
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Tell dimension of data
print('Dimension: ' + str(dataset.shape))

# Print first 20 columns of data
print(dataset.head(20))

# Give basic statistics of data
print(dataset.describe())

# Class distribution
print(dataset.groupby('class').size())

# Generate histograms for each row
dataset.hist()
# plt.show()

# Generate a mix of scatter plot and histograms to compare each dimension with one another
scatter_matrix(dataset)
# plt.show()

# Creating and evaluvating the models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))

results = []
names = []

seed = 7
scoring = 'accuracy'

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed, shuffle=True)
    # Apply preprocessing module to data to transform API when calculating the statistics of X_train
    # X_train = preprocessing.scale(X_train)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # print(msg)

'''
Terminal Output: 
LR: 0.966667 (0.040825)
KNN: 0.966667 (0.040825)
SVM: 0.975000 (0.038188)

SVM is the most accurate with the least standard deviation, but then again, it could be overfitting.
Further testing is needed to verify what is the best method for most accurate method
'''

# Test data with validation set and look at accuracy rates
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))

'''
LR
0.8666666666666667
KNN
0.9
SVM
0.8666666666666667

Most accurate on training data is KNN, thus KNN is the best option of all methods tested
'''
