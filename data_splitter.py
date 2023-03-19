#goofs up because it leaves the index when it saves

import pandas as pd
import numpy as np

a =  pd.read_csv('nli_for_simcse.csv')

numExamples = len(a)
numExamples

train = 8544
dev = 1101
test = 2210
total = train + dev + test
cnts = [train, dev, test]
fracs = [i/total for i in cnts]
fracs = [.70,.10,.20]
fracs
trainEnd = int(numExamples * fracs[0])
devEnd = int(numExamples * (fracs[1] + fracs[0]))
print(trainEnd, devEnd)

trainSet = a[0:trainEnd]
devSet = a[trainEnd:devEnd]
testSet = a[devEnd:]
sets = [trainSet, devSet, testSet]

names = ['nli_for_simcse-' + i + '.csv' for i in ['train', 'dev','test']]
for i in range(3):
    sets[i].to_csv(names[i])