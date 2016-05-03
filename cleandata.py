import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_data(datafile):
    if datafile not in ['quiz', 'data']:
        print('Enter "quiz" or "data" you moron')
        return []
    csv_file_obj = csv.reader(open('data/'+ datafile +'.csv', 'r'))
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

    #####
    # Clean the data
    #####

    # turn categorical data (strings) into integers.
    # also, find out which vectors are categorical

    # a vector of booleans which determines if the corresponding feature
    # in our data is categorical
    feature_types = []
    # for every feature except for the labels
    for j in range(len(data[0])-1):
        counter = 0
        # a dictionary containing all the attributes seen thus far
        seenWords = {}
        feature_types.append(type(data[0][j]))

        # if not categorical
        if not isinstance(data[0][j], str):
            # do scaling if float values
            if isinstance(data[0][j], float):
                ss = StandardScaler()
                # data[:,j] = ss.fit_transform(data[:,j].reshape(-1, 1).T)
                data[:,j] = ss.fit_transform(data[:,j])
            continue
        # mark the feature as being "categorical" for later use in the one hot encoder
        for i in range(len(data)):
            # for a new attribute
            if data[i][j] not in seenWords:
                seenWords.update({data[i][j]:counter})
                # replace the data with a number representing that attribute
                data[i][j]  = counter
                counter += 1
            else:
                # just use the number that represents a previously-seen attribute
                data[i][j]=seenWords[data[i][j]]
    categorical_features = [i for i in range(len(feature_types)) if feature_types[i] == type('s')]
    # TODO: although it may be unnecessary
    # use a line to fill in any NaNs in the data with the mean

    # do the one hot encoding
    # TODO: this makes everything into a float. May want to change this?
    enc = OneHotEncoder(categorical_features=categorical_features, dtype=np.int32)
    one_hot_data = enc.fit_transform(data).toarray()
    return one_hot_data

def store_data(datafile, data):
    filename = "data/" + datafile
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

            else:
                data[i][j]=seenWords[data[i][j]]
    np.savetxt(filename, data, delimiter=",",fmt="%02d")
'''
        creates prediction csv file for submission
'''
def store_csv(y_hat, filename):
    outfile = filename + ".csv"
    file = open(outfile, 'w+')
    file.write("Id,Prediction\n")
    for i, yi in enumerate(y_hat):
        file.write(str(i) + "," + str(yi) + "\n")
