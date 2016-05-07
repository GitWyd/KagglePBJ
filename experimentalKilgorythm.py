# Kaggle Competition COMS W4771
# PBJ
# Benjamin Lerner, Philippe Wyder (c)

import numpy
import time
import pickle
import getpass
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from cleandata_NoHot import get_data
from cleandata import store_data
from cleandata import store_csv

def main():
    '''
        FOR FUTURE REFERENCE:
        "train" refers to the data set which is trained
        "test" refers to the untrained portion of the data set on which the training is validated
        "quiz" refers to the unlabeled points which we attempt to label and then submit to kaggle
    '''
    start = time.time()
    # DO NOT MODIFY MAX_TRAIN_SIZE
    MAX_TRAIN_SIZE = 126838
    train_size = MAX_TRAIN_SIZE
    val_size = 20000

    print('Getting data...')
    data, quiz_data = get_data('data')
    X = data[:,0:-1]
    print("Number of features is: " + str(len(X[0])))
    y = [lbl for lbl in data[:,-1]]
    print('Received data, took this many seconds: ' + str(time.time() - start))
    # Training classifier

    # TODO: ExtraTreesClassifier

    clf1 = RandomForestClassifier(n_estimators=130, n_jobs= -1 if getpass.getuser() == 'ubuntu' else 1)
    #clf1 = RandomForestClassifier(      n_estimators=200,
    #                                    criterion='gini',
    #                                    max_depth=None,
    #                                    min_samples_split=2,
    #                                    min_samples_leaf=1,
    #                                    # min_weight_fraction_leaf=0.0001,
    #                                    max_features='auto',
    #                                    max_leaf_nodes=None,
    #                                    bootstrap=True,
    #                                    oob_score=False,
    #                                    n_jobs=-1,
    #                                    random_state=None,
    #                                    verbose=3,
    #                                    warm_start=False,
    #                                    class_weight=None
    #                              )
   # fit sub-classifiers
    clf1.fit(X,y)
    # print([estimator.tree_.max_depth for estimator in clf1.estimators_])
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
    # if train_size + val_size < MAX_TRAIN_SIZE:
    X_val = data[30000:60000,0:-1]
    y_val = [lbl for lbl in data[30000:60000,-1]]
    y_val_hat = clf1.predict(X_val)
    test_err = 1
    for yi, y_hati in zip(y_val, y_val_hat):
        test_err += (yi == y_hati)
    test_err /= X_val.shape[0]
    print("Test error: " + str(test_err))

    print("Beginning quiz validation...")
    X_test = quiz_data[:,:]
    y_test = [lbl for lbl in data[:,-1]]
    y_test_hat = clf1.predict(X_test)
    store_csv(y_test_hat, "experimental_prediction")
    end = time.time()
    duration = end - start
    print("Finished. Total duration: " + str(duration))

main()
