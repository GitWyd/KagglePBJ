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
from sklearn.decomposition import PCA


def main():
    data_size=120000 
    print("getting data")
    data, quiz_data = get_data('data')
    print("data got")
     
    results = []
    for p in range(50,200,5):
        errors= class_validator(expKilgorithm,p,data,data_size,5)
        print(p)
        print("Mean of errors:")
        print(np.mean(errors))
        results.append(np.mean(errors))
        print("\n")
    

    iteration=50
    for i in results:
        print(iteration)
        print(i)
        iteration+=5
    
def class_validator(classifier,parameter,data,data_size,p):
    """
        classifier must be take (trainX, trainY, valX,valY), data_size is number of
        of training points and p is number of partitions
    """

    data = data[:data_size]
    X = data[:,0:-1]

    #######
    # TEST PCA HERE
    ####
    pca = PCA()
    

    y = [lbl for lbl in data[:,-1]]
    #array of data partitions
    data_partitions = np.array_split(X,p)
    label_partitions = np.array_split(y,p)
    
    print(len(data_partitions[p-1]))
    print(len(label_partitions[p-1]))
    
    
    
    
    errors = []
    for i in range(len(data_partitions)):
        trainX=[]
        trainY=[]
        for j in range(p):
            if j!=i:
                trainX.extend(data_partitions[j])
                trainY.extend(label_partitions[j])
        
        

        valX = data_partitions[i]
        valY = label_partitions[i] 


        ###PCA TEST
        """
        pca.fit(trainX)
        trainX = pca.transform(trainX)
        valX = pca.transform(valX)
        """

        errors.append(classifier(trainX,trainY,valX,valY,parameter))
        
    return errors


def expKilgorithm(X,y,X_val, y_val,p):
    '''
        FOR FUTURE REFERENCE:
        "train" refers to the data set which is trained
        "test" refers to the untrained portion of the data set on which the training is validated
        "quiz" refers to the unlabeled points which we attempt to label and then submit to kaggle
    '''
    start = time.time()

    """
    
    X = data[train_start_idx:train_end_idx,0:-1]
    y = [lbl for lbl in data[train_start_idx:train_end_idx,-1]]
    """
    print('Received data, took this many seconds: ' + str(time.time() - start))
    # Training classifier

    # TODO: ExtraTreesClassifier

    clf1 = RandomForestClassifier(      n_estimators=p,
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
    train_err = 0
    for yi, y_hati in zip(y, y_hat):
        train_err += (yi != y_hati)
    train_err = float(train_err)/float(len(y))
    
    print("Train err: " + str(train_err))

    print("Beginning test validation...")
  
        
    y_val_hat = clf1.predict(X_val)
    test_err = 0
    for yi, y_hati in zip(y_val, y_val_hat):
        test_err += (yi != y_hati)
    print(len(y_val))
    print(len(y_val_hat))

    test_err = float(test_err)/float(len((y_val))) 
    print("Test error: " + str(test_err))
    
    return test_err


main()
