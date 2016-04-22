# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
def main():
    
    f = open("data.csv","r")
    l = f.readline()
    a=[]  
    d = np.genfromtxt("data.csv", delimiter = ",")
    print(d[:,47])
    print(d[:,48])
    x0 = []
    y0 = []
    y1 = []
    x1 = [] 
    for r in d:
        if r[52]==-1:
            x0.append(r[47])
            y0.append(r[48])
        else:
            x1.append(r[47])
            y1.append(r[48])
    plt.plot(x0,y0,"rx",x1,y1,"bo")
    plt.show()
 
main()