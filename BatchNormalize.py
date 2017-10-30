# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:05:01 2017

@author: thy1995
"""

from scipy.stats import rankdata
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import csv
from os.path import isfile, join
from os import listdir
from fileOP import writeRows
from resultOP import table_result
import os

def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)

#signature = "etf"
signature_l = ["ett", "etf","eft","eff","utf","uft","uff","utt"]
signature_l = ["ett"]
#signature_l = ["ett"]
direction = [True, False, False, True, True, True, False, False, False]
rank_all_list = []

combos =[]
for i in range(2, len(direction) + 1):
    combos.extend(combinations(range(1,len(direction) +1 ),i))

internalFolderSave = "D:\\CLS_lab\\AmericaFolder\\clustGUI-master\\real_data\\InternalRank\\"
internal_folder = "D:\\CLS_lab\\AmericaFolder\\clustGUI-master\\real_data\\internal - Copy\\"
makeFolder(internalFolderSave)
for signature in signature_l:

    savefolder = "D:\\CLS_lab\\AmericaFolder\\clustGUI-master\\tempD3\\" + signature + "\\"
    #makeFolder(savefolder)
    
    folder = "D:\\CLS_lab\\AmericaFolder\\clustGUI-master\\real_data\\Algorithm\\"
    
    folder = folder + signature + "\\"
    
    
    
    
    
    internal_file = [(internal_folder + f) for f in listdir(internal_folder) if isfile(join(internal_folder, f))]
    #internal_file = [i for i in internal_file if i.find(signature) != -1]
    realname = [t.split("\\")[-1].replace("internal.csv","") for t in internal_file]    

        
    
    header = [["Sil", "Db", "Xb", "Dunn", "CH", "I", "SD", "SDb_w", "CVNN"]]
    att = [['','NMI', "Adjusted Rand", "Accuracy", "Jaccard"]]
    
    header_1d = ["Sil", "Db", "Xb", "Dunn", "CH", "I", "SD", "SDb_w", "CVNN"]
    att_1d = ['NMI', "Adjusted Rand", "Accuracy", "Jaccard"]
    centroid = ['c2','c3','c4','c5','c6','c7','c8', 'c9', 'c10']
    
    #data_peak = np.recfromcsv(internal_file, delimiter = ',') # peak through data to see number of rows and cols
    num_cols = 10
    num_rows = 5
    data_internal_list  = [] # num_cols - 1 means skip label col
    data_external_list = []
    internal_row_c = 0

    removeKeywords = ['Kmeans', 'Spectral', 'Complete', 'Average', 'Ward']
    
    for internal_file_name in internal_file:
        data_peak = np.recfromcsv(internal_file_name, delimiter = ',')
        
        num_cols = len(data_peak[0])
        num_rows = len(data_peak)
         # num_cols - 1 means skip label col
        interval = 10
        
            
        data_internal  = np.zeros([num_rows+1, num_cols]).tolist()    
        with open(internal_file_name) as csvfile:
            row_index = 0
            reader= csv.reader(csvfile)
            for row in reader:
                for cols_index in range(num_cols):
                    data_internal[row_index][cols_index]= row[cols_index]
                row_index+=1
        
        data_internal = np.array(data_internal[1:])
        data_internal = (data_internal.T[:-1]).T
        data_internal_list.append(data_internal.astype(float))
    
    
    for dataset_index in range(len(internal_file)):
        print(internal_file[dataset_index])
        intern = data_internal_list[dataset_index]
        rank_list = np.zeros(9)
        for i in range(len(intern)):
            d = direction[i]
            temp = np.array(intern[i])
            if d == False:
                temp = -temp
            r = rankdata(temp)
            r  = r - (np.max(r) - 3)
            r = np.clip(r, a_min = 0, a_max = None)
            #rank_list = rank_list + rankdata(temp)
            rank_list = rank_list + r
#        intern = intern.tolist()
#        intern.append(rank_list.tolist())
#        to_export = table_result(intern,[['k' + str(i) for i in range(2, len(intern[0]) + 2 )]] ,[['','Sil', 'Db', 'Xb', 'Dunn', 'CH', "I", "SD", "SDb_w", "CVNN", "Rank"]])
#        name = internal_file[dataset_index].split(".")[0].split("\\")[-1]
#        writeRows(internalFolder + name + "internalR.csv" , to_export)
        intern_combination = []
        for c in combos:
            temp_c  = [intern[cc-1] * ((-1) ** (int(direction[cc - 1]) + 1)) for cc in c ]
            temp_cc  = [intern[cc-1] * ((-1) ** (int(direction[cc - 1]) + 1)) for cc in c ]
            temp_c  = np.transpose(temp_c)
            scaler = MinMaxScaler()
            scaler.fit(temp_c)
            temp_c  = np.transpose(scaler.transform(temp_c))

            temp_c = np.sum(temp_c, axis = 0)
            intern_combination.append(temp_c)
            
        intern = intern.tolist()
        intern.extend(intern_combination)
        str_col = ['','Sil', 'Db', 'Xb', 'Dunn', 'CH', "I", "SD", "SDb_w", "CVNN"]
        str_col.extend(['|'.join(map(str,a)) for a in combos])
        to_export = table_result(intern,[['k' + str(i) for i in range(2, len(intern[0]) + 2 )]] ,[str_col])
        name = internal_file[dataset_index].split(".")[0].split("\\")[-1]
        writeRows(internalFolderSave + name + "internalRC.csv" , to_export)