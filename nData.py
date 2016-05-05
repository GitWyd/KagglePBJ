# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:37:09 2016

@author: Philippe
"""
import csv
import numpy as np
from cleandata import get_data
import pickle

X_data, X_quiz = get_data('data')
print('data loaded...')
pickle.dump(, file, protocol=None, *, fix_imports=True)
np.savetxt("data/clean_data.csv", X_data, delimiter=",")
print('clean_data.csv has been saved')
np.savetext("data/clean_quiz.csv", X_quiz, delimiter=",")
print('clean_quiz.csv has been saved')
print('DONE')
