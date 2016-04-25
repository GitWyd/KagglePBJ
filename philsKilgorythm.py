# Kaggle Competition COMS W4771
# PBJ
# Philippe Wyder (c)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import VotingClassifier
import numpy
from cleandata import get_data
from cleandata import store_data
from cleandata import store_csv

def main():
    train_size = 1500
    data = get_data('data') 
    X = data[0:train_size,0:-1] 
    y = [lbl for lbl in data[0:train_size,-1]]
    print(X.shape)
    print(len(y))
    # Training classifier
    # declaring weights
    w = [1,1,1] 
    clf1 = DecisionTreeClassifier(max_depth=6)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = OrthogonalMatchingPursuit(n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute='auto')
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=w)
    clf1 = clf1.fit(X,y)
    clf2 = clf2.fit(X,y)
    clf3 = clf3.fit(X,y)
    clf4 = clf4.fit(X,y)
    eclf = eclf.fit(X,y)
    y_hat = eclf.predict(X) 
    test_err = 1
    for yi, y_hati in zip(y, y_hat):
        test_err += (yi == y_hati) 
    test_err /= train_size
    print("train: " + str(test_err))
    # validation data
    val_start = train_size
    val_end = train_size*2
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
    for yi, y_hati in zip(y_test, y_test_hat):
        test_err += (yi == y_hati) 
    test_err /= X_test.shape[0] 
    print("test: " + str(test_err))
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

