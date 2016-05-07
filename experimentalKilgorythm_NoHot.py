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
    
    '''
        FOR FUTURE REFERENCE:
        "train" refers to the data set which is trained
        "test" refers to the untrained portion of the data set on which the training is validated
        "quiz" refers to the unlabeled points which we attempt to label and then submit to kaggle
    '''
    # DO NOT MODIFY MAX_TRAIN_SIZE
    MAX_TRAIN_SIZE = 126838
    parameter = 100
    train_size = 100000
    val_size = MAX_TRAIN_SIZE-train_size

    print('Getting data...')
    data, quiz_data = get_data('data')
    X = data[0:train_size,0:-1]
    y = [lbl for lbl in data[0:train_size,-1]]
    print('Received data, took this many seconds: ')    
    # create validation set 
    val_start = train_size
    val_end = train_size+val_size
    # check to make sure we won't have an index out of bounds error
    if val_end <= MAX_TRAIN_SIZE:
        X_val = data[val_start:val_end,0:-1]
        y_val = [lbl for lbl in data[val_start:val_end,-1]]

    f = open('Analysis_forestOptimizer.csv', 'w+') 
    f.write('n_iter,alpha,average,score\n')
    for n_est in [90,100,115,140]:
        for crit in ['gini','entropy']:
            for max_dep in [80,100,150,200]:
                for warm_strt in  [True,False]:
                    f.write(str(n_est) +',' + str(crit) + ',' + str(max_dep) + ',' + str(warm_strt) + ','+ str(classifier(X, y, X_val, y_val, n_est, crit, max_dep, warm_strt)) + '\n')
 
    f.close() 

def classifier(X, y, X_val, y_val, n_est, crit, max_dep, warm_strt):
    # Training classifier
    start = time.time()

    # TODO: ExtraTreesClassifier

    clf1 = RandomForestClassifier(      n_estimators= n_est,
                                        criterion=crit,
                                        max_depth=max_dep,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        # min_weight_fraction_leaf=0.0001,
                                        max_features='auto',
                                        max_leaf_nodes=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=-1,
                                        random_state=None,
                                        verbose=False,
                                        warm_start=warm_strt,
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
    train_err /= X.shape[0]
    print("Train err: " + str(train_err))

    print("Beginning test validation...")
    y_val_hat = clf1.predict(X_val)
    test_err = 1
    for yi, y_hati in zip(y_val, y_val_hat):
        test_err += (yi == y_hati)
    test_err /= X_val.shape[0]
    print("Test error: " + str(test_err))
    
    '''
    print("Beginning quiz validation...")
    X_test = quiz_data[:,:]
    y_test = [lbl for lbl in data[:,-1]]
    y_test_hat = clf1.predict(X_test)
    store_csv(y_test_hat, "experimental_NoHot_prediction")
    '''
    end = time.time()
    duration = end - start
    return test_err
    print("Finished. Total duration: " + str(duration))

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
