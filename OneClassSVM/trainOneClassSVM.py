import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM

import _pickle as cPickle

# Data
num_train_data = 100000
num_joint = 7

train_data = genfromtxt('./data/backward_residual.csv')
train_idx = np.random.randint(train_data.shape[0], size=num_train_data)

clf = OneClassSVM(gamma='auto', nu=0.00001).fit(train_data[train_idx])

# save the classifier
with open('ocsvm_residual.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    