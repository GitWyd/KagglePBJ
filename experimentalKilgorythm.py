# Kaggle Competition COMS W4771
# PBJ
# Benjamin Lerner, Philippe Wyder (c)

import numpy
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
    start = time.time()
    MAX_TRAIN_SIZE = 126838
    train_size = MAX_TRAIN_SIZE
    val_size = 32000
    data, test_data = get_data('data')
    X = data[0:train_size,0:-1]
    y = [lbl for lbl in data[0:train_size,-1]]
    print(X.shape)
    print(len(y))

#    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#    X = scaler.fit_transform(X)
    # Training classifier

    # TODO: ExtraTreesClassifier

    clf1 = RandomForestClassifier(      n_estimators=100,
                                        criterion='entropy',
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
    test_err = 1
    for yi, y_hati in zip(y, y_hat):
        test_err += (yi == y_hati)
    test_err /= train_size
    print("train: " + str(test_err))

    # validation data - calculate valdiation error
    # val_start = train_size
    # val_end = train_size + val_size

    # get validation data set
    # TODO: put this back in
    # if MAX_TRAIN_SIZE - train_size > val_size:
    print("Beginning test validation...")
    X_val = data[:val_size,0:-1]
    y_val = [lbl for lbl in data[:val_size,-1]]
    y_val_hat = clf1.predict(X_val)
    test_err = 1
    for yi, y_hati in zip(y_val, y_val_hat):
        test_err += (yi == y_hati)
    test_err /= X_val.shape[0]
    print("val: " + str(test_err))

    #quiz data
    print("Beginning quiz validation...")
    # test_data = get_data('quiz')
    X_test = test_data[:,:]
    print(X_test.shape)
    y_test = [lbl for lbl in data[:,-1]]
    y_test_hat = clf1.predict(X_test)
    store_csv(y_test_hat, "experimental_prediction")
    end = time.time()
    duration = end - start
    print("Took this many seconds: " + str(duration))

main()
'''
    # update weights with validation data sets
    val_start = train_size
    val_end = train_size*2
    num_validation_rnds = 4
    for i in range(num_validation_rnds):
        # get validation data set
        X_val = data[val_start:val_end,0:-1]
        y_val = [lbl for lbl in data[val_start:val_end,-1]]
        # score separate classifiers on validation set
        clf1_score = clf1.score(X_val,y_val)
        clf2_score = clf2.score(X_val,y_val)
        clf3_score = clf3.score(X_val,y_val)
        # re-weigh the classifiers
        total = clf1_score + clf2_score + clf3_score
        w = [(clf1_score + w[0])/total, (clf2_score + w[1])/total, (clf3_score + w[2])/total]
        print(w)
        clf1 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=w)
        clf1 = clf1.fit(X_val,y_val)
        # select new validation data
        val_start = val_end
        val_end = val_end + train_size
'''
