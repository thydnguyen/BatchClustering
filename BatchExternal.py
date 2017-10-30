# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:01:55 2017

@author: thy1995
"""

import ExterValid
from os.path import isfile, join
from os import listdir


truthFolder = "D:\\CLS_lab\\codeTest\\batchSynthetic\\labelRevisedM\\"
labelFolder = "D:\\CLS_lab\\temp\\clustering\\"
externalFolder = "D:\\CLS_lab\\temp\\external2\\"
truthFiles = [(truthFolder + f) for f in listdir(truthFolder) if isfile(join(truthFolder, f))]
labelFiles = [(labelFolder + f) for f in listdir(labelFolder) if isfile(join(labelFolder, f))]

counter = 0
for tr in truthFiles:
    print(counter)
    target= tr.split("\\")[-1].split(".csv")[0]
    targets = [i for i in labelFiles if i.find(target) != -1]
    name = [t.split(".")[0].split("\\")[-1] for t in targets]
    outputTargets = [ externalFolder + n + "external.csv" for n in name]
    for i in range(len(targets)):
        ExterValid.generate_report(tr, targets[i], outputTargets[i])
    counter = counter + 1