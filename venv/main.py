import csv

from sklearn.metrics import accuracy_score

from CFS.CFS import cfs
import numpy as np
from sklearn.naive_bayes import GaussianNB

with open('names.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)

n = len(dataForAnalys)
p = len(dataForAnalys[0])

data = np.array(dataForAnalys)

Y = data[:, 10]
Y = np.float64(Y)

predictX = np.zeros((n, p - 1))
for k in range(p - 1):
    predictX[:, k] = data[:, k]
S = cfs(predictX, Y)

newPredictX = np.zeros((n, len(S)))
for k in range(len(S)):
    newPredictX[:, k] = data[:, S[k]]


model = GaussianNB()
model.fit(newPredictX, Y)

predicted = model.predict(newPredictX)

print(accuracy_score(predicted, Y))