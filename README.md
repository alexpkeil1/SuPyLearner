SuPyLearner
===========

An implementation of the SuperLearner algorithm in Python based on scikit-learn.

Now updated for Python 3.5
# Installation
#### OSX
Run the following in terminal

    git clone https://github.com/alexpkeil1/supylearner.git
    cd supylearner
    python setup.py install

Example (from examples/typical_usage.py)

```python
# typical usage
import supylearner as sl
from sklearn import datasets, svm, linear_model, neighbors, svm
import numpy as np

# generate dataset
np.random.seed(100)
X, y = datasets.make_friedman1(1000)

ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(l1_ratio = .1)
ridge = linear_model.RidgeCV()
lars = linear_model.LarsCV()
lasso = linear_model.LassoCV()
nn = neighbors.KNeighborsRegressor()
svm1 = svm.SVR(kernel = 'rbf') 
svm2 = svm.SVR(kernel = 'poly')
lib = [ols, elnet, ridge,lars, lasso, nn, svm1, svm2]
libnames = ["OLS", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly"]

sl_inst = sl.SuperLearner(lib, libnames, loss = "L2")
sl_inst.fit(X, y)

sl_inst.summarize()
```

```
    Cross-validated risk estimates for each estimator in the library:
    [['OLS' '5.889258599506168']
    ['ElasticNet' '6.0149186311686185']
    ['Ridge' '5.8892340442419115']
    ['LARS' '5.869122063410273']
    ['LASSO' '5.866767295197982']
    ['kNN' '7.037900242493755']
    ['SVM rbf' '6.242369357877353']
    ['SVM poly' '15.520506952085686']]

    Coefficients:
    [['OLS' '0.636769614909252']
    ['ElasticNet' '0.0']
    ['Ridge' '0.0']
    ['LARS' '0.0']
    ['LASSO' '0.0']
    ['kNN' '0.3632303850907481']
    ['SVM rbf' '0.0']
    ['SVM poly' '0.0']]
    (Not cross-valided) estimated risk for SL: 5.33523373261
```

```python
sl.cv_superlearner(sl_inst, X, y, K = 5)
```
```
    Cross-validated risk estimates for each estimator in the library and SuperLearner:
    [['OLS' '5.889258599506169']
    ['ElasticNet' '6.014918631168618']
    ['Ridge' '5.8892340442419115']
    ['LARS' '5.869122063410274']
    ['LASSO' '5.866767295197983']
    ['kNN' '7.037900242493755']
    ['SVM rbf' '6.242369357877353']
    ['SVM poly' '15.520506952085688']
    ['SuperLearner' '5.340871446622769']]
```
