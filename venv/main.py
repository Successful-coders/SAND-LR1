import pandas as pd
import numpy as np
import csv
from CFS.CFS import cfs

with open('names.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)

n = len(dataForAnalys)
p = len(dataForAnalys[0])

data = np.array(dataForAnalys)

Y = data[:, 10]

predictX = np.zeros((n, p - 1))
for k in range(p - 1):
    predictX[:, k] = data[:, k]
S = cfs(predictX, Y)

newPredictX = np.zeros((n, len(S)))
for k in range(len(S)):
    newPredictX[:, k] = data[:, S[k]]
print(newPredictX)