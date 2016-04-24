# Kaggle Competition COMS W4771
# PBJ
# Philippe Wyder (c)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import numpy
from cleandata import get_data
from cleandata import store_data

def main():
    data = get_data('data') 
    X = data[0:1000,0:-1] 
    y = [lbl for lbl in data[0:1000,-1]]
    print(X.shape)
    print(len(y))
    # Training classifier
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
    clf1 = clf1.fit(X,y)
    clf2 = clf2.fit(X,y)
    clf3 = clf3.fit(X,y)    
    eclf = eclf.fit(X,y)
    y_hat = eclf.predict(X) 
    print(y_hat) 
    #store_data("killerFile",y_hat)
        
main()

