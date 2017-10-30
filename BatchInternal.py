# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:03:58 2017

@author: thy1995
"""

from os.path import isfile, join
from os import listdir
import numpy as np
import csv

from fileOP import writeRows
import internal_validation
from sklearn.metrics import silhouette_score
from resultOP import table_result

dataFolder = "D:\\CLS_lab\\codeTest\\batchSynthetic\\dataRevisedM\\"
labelFolder = "D:\\CLS_lab\\temp\\clustering\\"
internalFolder = "D:\\CLS_lab\\temp\\internal2\\"
import os.path

dataFiles = [(dataFolder + f) for f in listdir(dataFolder) if isfile(join(dataFolder, f))]
labelFiles = [(labelFolder + f) for f in listdir(labelFolder) if isfile(join(labelFolder, f))]

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
    
    target= data_file_name.split("\\")[-1].split(".csv")[0]
    targets = [i for i in labelFiles if i.find(target) != -1]
    
    scatL = []
    distL = []
    comL = []
    sepL = []
    for label_file_name  in targets:
        print("current label", label_file_name)
        name = label_file_name.split(".")[0].split("\\")[-1]
        exist = os.path.exists(internalFolder + name + "internal.csv")
        if exist:
            continue
        
        data_peak = np.recfromcsv(label_file_name, delimiter = ',') # peak through data to see number of rows and cols

        num_cols = len(data_peak[0])
        num_rows = len(data_peak)
        label  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    
    
        with open(label_file_name) as csvfile:
            row_index = 0
            reader= csv.reader(csvfile)
            for row in reader:
                for cols_index in range(num_cols):
                    label[row_index][cols_index]= row[cols_index]
                row_index+=1
        
        label = label.T
        for d_column in label:
            num_k = np.unique(d_column)
            inter_index = internal_validation.internalIndex(len(num_k))
            
            scat , dis = inter_index.SD_valid(data, d_column)
            com , sep = inter_index.CVNN(data, d_column)
            scatL.append(scat)
            distL.append(dis)
            comL.append(com)
            sepL.append(sep)
        result_over_k = []
        for i in range(len(label)):
            d_column = label[i]
            num_k = np.unique(d_column)
            result = []
            inter_index = internal_validation.internalIndex(len(num_k))
            result.append(silhouette_score(data, d_column, metric = 'euclidean'))
            result.append(inter_index.dbi(data, d_column))
            result.append(inter_index.xie_benie(data, d_column))
            result.append(inter_index.dunn(data, d_column))
            result.append(inter_index.CH(data, d_column))
            result.append(inter_index.I(data, d_column))
            result.append(inter_index.SD_valid_n(scatL, distL, i))
            result.append(inter_index.SDbw(data, d_column))
            result.append(inter_index.CVNN_n(comL, sepL, i))
            
            result_over_k.append(result)
        to_export = np.array(result_over_k).T
        to_export = table_result(to_export,[['k' + str(i) for i in range(2, len(to_export[0]) + 2 )]] ,[['','Sil', 'Db', 'Xb', 'Dunn', 'CH', "I", "SD", "SDb_w", "CVNN"]])
        
        name = label_file_name.split(".")[0].split("\\")[-1]
        writeRows(internalFolder + name + "internal.csv" , to_export)
    counter = counter + 1