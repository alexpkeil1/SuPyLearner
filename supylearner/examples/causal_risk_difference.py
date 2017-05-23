#!/usr/bin/env python 
#######################################################################################################################
# Author: Alex Keil
# Program: causal_risk_difference.py
# Language: python
# Date: Tuesday, May 23, 2017 at 3:16:39 PM
# Project: supylearner examples
# Tasks: simulate data and perform causal risk analysis repeatedly, summarize over 100 iterations
# Description: causal risk difference using g-computation estimator for causal risk difference
# relevant manuscript: 
#  A. P. Keil, J. K. Edwards, D. B. Richardson, A. I. Naimi, and S. R. Cole. 
#   The parametric g-formula for time-to-event data: Intuition and a worked example. 
#   Epidemiology, 25(6):889–897, 2014.
# Commands: chmod u+x causal_risk_difference.py ./causal_risk_difference.py
# Released under the GNU General Public License: http://www.gnu.org/copyleft/gpl.html
# warning: takes a while to run
######################################################################################################################

# 
# CF: 
#
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import supylearner as sl

def expit(mu):
    return 1/(1+np.exp(-mu))


def dgm(n=300, truerd=0.1):
    py0_true = np.random.uniform(size=n)*0.1 + 0.4  
    l = np.random.binomial(1, expit(1 - py0_true), n)
    c = np.random.normal(py0_true, 1, n)
    c2 =  np.random.normal(py0_true, .3, n)
    x = np.random.binomial(1, expit(1.5 - 2*l - c - c2), n)
    py = py0_true + truerd*x #*true risk difference per unit exposure
    y = np.random.binomial(1, py, n)
    return l,c,c2,x,y

def fitall(X,y):
    m1 =  RandomForestClassifier(n_estimators=500)
    m2 =  LogisticRegression()
    m3 =  SVC(probability=True)
    lib = [m1, m2, m3]
    m4 = sl.SuperLearner(lib, loss = "nloglik")
    return m1.fit(X,y), m2.fit(X,y), m3.fit(X,y), m4.fit(X, y)


def predall(X,y):
    X1, X0 = X.copy(), X.copy()
    X1[:,1] = np.ones_like(y)
    X0[:,1] = np.zeros_like(y)
    f1, f2, f3, f4 = fitall(X,y)
    rd = [0]*4
    for i,f in enumerate([f1, f2, f3]):
        rd[i] = np.mean(f.predict_proba(X1)[:,1] - f.predict_proba(X0)[:,1])
    rd[3] = np.mean(f4.predict(X1) - f4.predict(X0))
    return rd


if __name__ == "__main__":
    iter = 100
    res = np.ndarray(shape=(iter,4))
    
    for iter in range(iter):
        l,c,c2,x,y = dgm(300, truerd=0.1)
        X = np.column_stack((np.ones_like(l), x, l,c,c2))
        res[iter,:] = predall(X,y)
    
        
    estimators = ['Random Forest', 'Logistic regression', 'Support vector classifier', 'SuperLearner']
    summary = [np.mean(res[:,i]) for i in range(4)]
    print("# "+"-"*60)
    print("# {:32}| {}".format("Estimator", "E(Y(1) - Y(0))"))
    print("# "+"-"*60)
    for i,s in enumerate(summary) :
        print('# {:32}| {:.4f}'.format(estimators[i], s))
    print("# "+"-"*60)
