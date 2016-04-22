import csv
import numpy as np

csv_file_obj = csv.reader(open('data/data.csv', 'r'))
header = next(csv_file_obj)
data = np.array([row for row in csv_file_obj])
print(data)
