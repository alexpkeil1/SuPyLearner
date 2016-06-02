SuPyLearner
===========

An implementation of the SuperLearner algorithm in Python based on scikit-learn.


Example (from examples/typical_usage.py)

```{python}
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

sl.cv_superlearner(sl_inst, X, y, K = 5)
```

