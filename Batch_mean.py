# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:29:36 2017

@author: thy1995
"""

import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from fileOP import writeRows

param = 250
folder = "\\\\EGR-1L11QD2\\CLS_lab\\codeTest\\fwl_project\\graph\\48batch\\" + str(param) + "\\"


extension = ".csv"
skip = 10
l_ex = len(extension)
onlyfiles = [(folder + f) for f in listdir(folder) if isfile(join(folder, f))]
checkpoints =  [a for a in onlyfiles if a[-l_ex:] == extension]
big_matrix = []
for ckpt in checkpoints:
    data_peak = np.recfromcsv(ckpt, delimiter = ',') # peak through data to see number of rows and cols
    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    with open(ckpt) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index]= row[cols_index]
            row_index+=1
    data = np.squeeze(data)
    data_skip = [data[i] for i in range(len(data)) if i % param > skip]
    big_matrix.append([ckpt, np.mean(data), np.mean(data_skip)])

writeRows(folder + str(param) + "stats.csv", big_matrix)

