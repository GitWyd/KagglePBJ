# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
def main():
    
   
    d = np.genfromtxt("data/data.csv", delimiter = ",", skip_header=1)
  
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
    plt.plot(x0,y0,"rx",x1,y1,"b.")
    plt.show()
 
main()