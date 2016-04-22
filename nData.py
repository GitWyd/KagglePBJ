# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:37:09 2016

@author: Jesse
"""

import csv
import numpy as np

csv_file_obj = csv.reader(open('data/data.csv', 'r'))
header = next(csv_file_obj)
# doesn't seem like it does anything
# convert the data
raw_data = []
for row in csv_file_obj:
    data_row = []
    for element in row:
        # element can be a float, int, string, or None.
        # the float() cast will work on both integers and floats. However the int()
        # cast will only work on ints; if element is a float, it will fail. So,
        # we attempt to cast element as an int:
        try:
            data = int(element)
        except ValueError:
            # if that didn't work, then the element may yet be a float
            try:
                data = float(element)
            # if that still didn't work, then the data is either a string or None.
            # TODO: possibly do something about None-esque values, deleting rows
            # or using averages instead
            except ValueError:
                data = element



        data_row.append(data)
    raw_data.append(data_row)
data = np.array(raw_data, dtype=object)
# data = np.array([row for row in csv_file_obj])

print(data)
for j in range(len(data[0])):
    counter = 0
    seenWords = {}
    if not isinstance(data[0][j], str):
        continue
    for i in range(len(data)):
        if data[i][j] not in seenWords:
            #print(data[i][j])
            seenWords.update({data[i][j]:counter})
            
            data[i][j]  = counter
            counter += 1
            
            #print(data[i][j])
        else:
            data[i][j]=seenWords[data[i][j]]
print(data)

#fnew = open("numData.csv","w")
