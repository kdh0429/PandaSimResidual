import numpy as np
from numpy import genfromtxt

from sklearn.svm import OneClassSVM

import _pickle as cPickle

# Data
num_train_data = 290000
num_joint = 7

train_data = genfromtxt('./data/TrainingData.csv', delimiter=',')
train_idx = np.random.randint(train_data.shape[0], size=num_train_data)

clf = OneClassSVM(gamma='scale', nu=0.000001).fit(train_data[train_idx])

# save the classifier
with open('./model/ocsvm_residual.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    

print("Training Finished!")