# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:37:09 2016

@author: Philippe
"""
from cleandata import get_data
import csv
import numpy as np

X_data, X_quiz = get_data('data')
print('data loaded...'
np.savetxt("data/clean_data.csv", X_data, delimiter=",")
print('clean_data.csv has been saved')
np.savetext("data/clean_quiz.csv", X_quiz, delimiter=",")
print('clean_quiz.csv has been saved')
print('DONE')
