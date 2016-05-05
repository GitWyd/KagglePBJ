# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:25:47 2016

@author: Jesse
"""

# Kaggle Competition COMS W4771
# PBJ
# Benjamin Lerner, Philippe Wyder (c)

import numpy as np
import time
import pickle
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from cleandata import get_data
from cleandata import store_data
from cleandata import store_csv

def main():
    
    errors= class_validator(expKilgorithm,5)
    print(errors)
    
def class_validator(classifier, p):
    """
        classifier must be take (trainX, trainY, valX,valY) and p is number 
        of partitions
    """
    data, quiz_data = get_data('data')
    data = data[:10000]

    X = data[:,0:-1]
    y = [lbl for lbl in data[:,-1]]
    #array of data partitions
    data_partitions = np.array_split(X,p)
    label_partitions = np.array_split(y,p)
    
    print(len(data_partitions[p-1]))
    print(len(label_partitions[p-1]))
    
    
    
    trainX=[]
    trainY=[]
    
    errors = []
    for i in len(data_partitions):
        for j in range(p):
            if j!=i:
                trainX.extend(data_partitions[j])
                trainY.extend(label_partitions[j])
        
        errors.append(classifier(trainX,trainY,data_partitions[i],label_partitions[i]))
        
    return errors


def expKilgorithm(X,y,X_val, y_val):
    '''
        FOR FUTURE REFERENCE:
        "train" refers to the data set which is trained
        "test" refers to the untrained portion of the data set on which the training is validated
        "quiz" refers to the unlabeled points which we attempt to label and then submit to kaggle
    '''
    start = time.time()
    # DO NOT MODIFY MAX_TRAIN_SIZE
    MAX_TRAIN_SIZE = 126838
    train_size = 100000
    val_size = 20000

    print('Getting data...')
    data, quiz_data = get_data('data')
    """
    
    X = data[train_start_idx:train_end_idx,0:-1]
    y = [lbl for lbl in data[train_start_idx:train_end_idx,-1]]
    """
    print('Received data, took this many seconds: ' + str(time.time() - start))
    # Training classifier

    # TODO: ExtraTreesClassifier

    clf1 = RandomForestClassifier(      n_estimators=100,
                                        criterion='gini',
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        # min_weight_fraction_leaf=0.0001,
                                        max_features='auto',
                                        max_leaf_nodes=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=-1,
                                        random_state=None,
                                        verbose=3,
                                        warm_start=False,
                                        class_weight=None
                                  )
   # fit sub-classifiers
    clf1.fit(X,y)
    # pickle.dump(clf1, open('experimental_classifier.pickle', 'wb'))

    # fit voting classifier

    # predict & calculate training error
    y_hat = clf1.predict(X)
    train_err = 1
    for yi, y_hati in zip(y, y_hat):
        train_err += (yi == y_hati)
    train_err /= train_size
    print("Train err: " + str(train_err))

    print("Beginning test validation...")
    # check to make sure we won't have an index out of bounds error
    if train_size + val_size < MAX_TRAIN_SIZE:
        """
        X_val = data[test_start_idx:test_end_idx,0:-1]
        y_val = [lbl for lbl in data[test_start_idx:test_end_idx,-1]]
        """
        
        y_val_hat = clf1.predict(X_val)
        test_err = 1
        for yi, y_hati in zip(y_val, y_val_hat):
            test_err += (yi == y_hati)
        test_err /= X_val.shape[0]
        print("Test error: " + str(test_err))
    
    return test_err


main()
