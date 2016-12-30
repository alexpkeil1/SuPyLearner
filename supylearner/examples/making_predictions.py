#not working
#!/usr/bin/env python
import supylearner as sl
from sklearn import datasets, svm, linear_model, neighbors, svm
import numpy as np
import statsmodels.api as sm

# generate dataset
np.random.seed(200)
N = 200
X = np.column_stack((np.ones(N), np.random.normal(0, 1, [N, 4])))
X1 = X.copy()
X0 = X.copy()
X1[:,1] = np.ones(N)
X0[:,1] = np.zeros(N)
d = np.column_stack((X, X[:,1:]*X[:,1:], X[:,1:]*X[:,1:]*X[:,1:]))
d1 = np.column_stack((X1, X1[:,1:]*X1[:,1:], X1[:,1:]*X1[:,1:]*X1[:,1:]))
d0 = np.column_stack((X0, X0[:,1:]*X0[:,1:], X0[:,1:]*X0[:,1:]*X0[:,1:]))
beta = np.random.normal(0, .5, d.shape[1])
beta[1] = 3
beta[5] = -2
mu = d.dot(beta)
mu1 = d1.dot(beta)
mu0 = d0.dot(beta)
eps = np.random.normal(0, 3, N)
y = mu + eps
y1 = mu1 + eps
y0 = mu0 + eps



ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(l1_ratio = .1)
ridge = linear_model.RidgeCV()
lasso = linear_model.LassoCV()
nn = neighbors.KNeighborsRegressor()
svm1 = svm.SVR(kernel = 'rbf') 
svm2 = svm.SVR(kernel = 'poly')
lib = [ols, elnet, ridge, lasso, nn, svm1, svm2]
libnames = ["OLS", "ElasticNet", "Ridge", "LASSO", "kNN", "SVM rbf", "SVM poly"]

sl_inst = sl.SuperLearner(lib, libnames, loss = "L2")
sl_inst.fit(X, y)
sl_inst.summarize()
cvsl_inst = sl.cv_superlearner(sl_inst, X, y, K = 5)

# more features
sl_inst2 = sl.SuperLearner(lib, libnames, loss = "L2")
sl_inst2.fit(d, y)
sl_inst2.summarize()
cvsl_inst2 = sl.cv_superlearner(sl_inst2, d, y, K = 5)


#  Coefficients:
#  [['OLS' '0.7864245846152594']
#   ['ElasticNet' '0.1511272799810129']
#   ['Ridge' '0.0']
#   ['LASSO' '0.0']
#   ['kNN' '0.062448135403727664']
#   ['SVM rbf' '0.0']
#   ['SVM poly' '0.0']]
#  
#  (Not cross-valided) estimated risk for SL: 7.98286994809
#  Cross-validated risk estimates for each estimator in the library and SuperLearner:
#  [['OLS' '8.055971271872655']
#   ['ElasticNet' '8.337953500659625']
#   ['Ridge' '8.077946327526314']
#   ['LASSO' '8.219156455972458']
#   ['kNN' '17.296243324196585']
#   ['SVM rbf' '37.39849752328567']
#   ['SVM poly' '254.7810291161271']
#   ['SuperLearner' '8.209745064797863']]


#ols, knn only
olsfit2 = sm.OLS(y,d).fit()
knnfit2 = nn.fit(d,y)

print('\n\n')
print('-'*7, ' Effect estimates ', '-'*7)
print('-'*7, '  E(Y(1) - Y(0))  ', '-'*7)
print('-'*34)
print('Data generating mechanism | {:.3f}|'.format(np.mean(y1-y0)))
print('-'*34)
print('SuperLearner only         | {:.3f}|'.format(np.mean(sl_inst2.predict(d1) - sl_inst2.predict(d0))))
print('OLS only                  | {:.3f}|'.format(np.mean(olsfit2.predict(d1) - olsfit2.predict(d0))))
print('kNN only                  | {:.3f}|'.format(np.mean(knnfit2.predict(d1) - knnfit2.predict(d0))))
print('-'*34)



# -------  Effect estimates  -------
# -------   E(Y(1) - Y(0))   -------
# ----------------------------------
# Data generating mechanism | 1.743|
# ----------------------------------
# SuperLearner only         | 1.587|
# OLS only                  | 1.718|
# kNN only                  | 0.695|
# ----------------------------------

