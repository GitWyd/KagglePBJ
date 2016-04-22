import csv
import numpy as np

csv_file_obj = csv.reader(open('data/data.csv', 'r'))
header = next(csv_file_obj)
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
            pass
        # if that didn't work, then the element may yet be a float
        try:
            data = float(element)
        # if that still didn't work, then the data must be either a string or None.
        # TODO: possibly do something about None-esque values, either deleting rows
        # or using averages instead
        except ValueError:
            data = element

        data_row.append(data)
    raw_data.append(data_row)

data = np.array(raw_data)
print(data)
