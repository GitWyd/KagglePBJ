import csv
import numpy as np

def get_data(datafile):
    if datafile not in ['quiz', 'data']:
        print('Enter "quiz" or "data" you moron')
        return []
    csv_file_obj = csv.reader(open('data/'+datafile+'.csv', 'r'))
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
                except ValueError:
                    data = element



            data_row.append(data)
        raw_data.append(data_row)
    data = np.array(raw_data, dtype=object)

    # clean the data
    for j in range(len(data[0])):
        counter = 0
        seenWords = {}
        if not isinstance(data[0][j], str):
            continue
        for i in range(len(data)):
            if data[i][j] not in seenWords:
                seenWords.update({data[i][j]:counter})

                data[i][j]  = counter
                counter += 1
            else:
                data[i][j]=seenWords[data[i][j]]
    return data
