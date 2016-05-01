# Kaggle Competition COMS W4771
# PBJ
# Philippe Wyder (c)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import numpy
from cleandata import get_data
from cleandata import store_data
from cleandata import store_csv

def main():
    train_size = 5000
    val_size = 32000
    data = get_data('data') 
    X = data[0:train_size,0:-1] 
    y = [lbl for lbl in data[0:train_size,-1]]
    print(X.shape)
    print(len(y))
    # Training classifier
    clf1 = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf2 = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    
    # fit sub-classifiers
    clf1.fit(X, y) 
    clf2.fit(X,y)
    # fit voting classifier
    
    # predict & calculate training error
    y_hat = clf1.predict(X) 
    test_err = 1
    for yi, y_hati in zip(y, y_hat):
        test_err += (yi == y_hati) 
    test_err /= train_size
    print("train: " + str(test_err))

    # validation data - calculate valdiation error
    val_start = train_size
    val_end = train_size + val_size
    
    # get validation data set
    X_val = data[val_start:val_end,0:-1] 
    y_val = [lbl for lbl in data[val_start:val_end,-1]]
    y_val_hat = eclf.predict(X_val) 
    test_err = 1
    for yi, y_hati in zip(y_val, y_val_hat):
        test_err += (yi == y_hati) 
    test_err /= X_val.shape[0]
    print("val: " + str(test_err)) 

    #quiz data    
    test_data = get_data('quiz')
    X_test = data[:,0:-1] 
    print(X_test.shape)
    y_test = [lbl for lbl in data[:,-1]]
    y_test_hat = eclf.predict(X_test) 
    test_err = 1
#    for yi, y_hati in zip(y_test, y_test_hat):
#        test_err += (yi == y_hati) 
#    test_err /= X_test.shape[0] 
#    print("test: " + str(test_err))
    store_csv(y_test_hat, "killgorythm_prediciton")  
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
        eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=w)
        eclf = eclf.fit(X_val,y_val)
        # select new validation data
        val_start = val_end
        val_end = val_end + train_size 
''' 

