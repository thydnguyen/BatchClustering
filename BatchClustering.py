# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:01:00 2017

@author: thy1995
"""

from os.path import isfile, join
from os import listdir
import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from fileOP import writeRows

#
#Clustering
dataFolder = "D:\\CLS_lab\\codeTest\\batchSynthetic\\dataRevisedM\\"
labelFolder = "D:\\CLS_lab\\temp\\clustering\\"
dataFiles = [(dataFolder + f) for f in listdir(dataFolder) if isfile(join(dataFolder, f))]
counter = 0
for data_file_name in dataFiles:
    print(counter)
    data_peak = np.recfromcsv(data_file_name, delimiter = ',') # peak through data to see number of rows and cols

    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    
    
    with open(data_file_name) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index]= row[cols_index]
            row_index+=1
    
    k_result = []
    s_result = []
    a_result = []
    c_result = []
    w_result = []
    
    for k in range(2, 11):
        k_result.append(KMeans(init='k-means++', n_clusters = k, n_init=10, max_iter = 1000).fit(data).labels_)
        s_result.append(SpectralClustering(n_clusters = k, affinity = "nearest_neighbors", n_neighbors= 15, n_init = 100 ).fit(data).labels_)
        a_result.append(AgglomerativeClustering(linkage='average', n_clusters = k).fit(data).labels_)
        c_result.append(AgglomerativeClustering(linkage='complete', n_clusters = k).fit(data).labels_)
        w_result.append(AgglomerativeClustering(linkage='ward', n_clusters = k).fit(data).labels_)
    
    k_result = np.array(k_result).T + 1
    s_result = np.array(s_result).T + 1
    a_result = np.array(a_result).T + 1 
    c_result = np.array(c_result).T + 1
    w_result = np.array(w_result).T + 1
    
    name = data_file_name.split(".")[0].split("\\")[-1]
    writeRows(labelFolder + name + "Kmeans.csv", k_result)
    writeRows(labelFolder + name + "Spectral.csv", s_result)
    writeRows(labelFolder + name + "Average.csv", a_result)
    writeRows(labelFolder + name + "Complete.csv", c_result)
    writeRows(labelFolder + name + "Ward.csv", w_result)
    counter = counter + 1