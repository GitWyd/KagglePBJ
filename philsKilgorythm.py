# Kaggle Competition COMS W4771
# PBJ
# Philippe Wyder (c)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import numpy

def main():
    X = loadTrainingData()
    Y = loadTrainingLabels()
    
    # Training classifier
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
    clf1 = clf1.fit(X,y)
    clf2 = clf2.fit(X,y)
    clf3 = clf3.fit(X,y)    
    eclf = eclf.fit(X,y)

def loadTrainingData():
    


if __name__ == "__main__":
    main()

