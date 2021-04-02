import pandas as pd
import numpy as np
import csv


def calcI(i, j, K, M, data):
    if i < K - 1 and j < M - 1:
        I = 0
        for q in range(i + 1, K):
            for r in range(j + 1, M):
                I += data[q][r]
        return I
    else:
        return 0


def calcIV(i, j, K, M, data):
    if i < K - 1 and j > 0:
        IV = 0
        for q in range(i + 1, K):
            for r in range(0, j - 1):
                IV += data[q][r]
        return IV
    else:
        return 0


def calcII(i, j, K, M, data):
    if i > 0 and j < M - 1:
        II = 0
        for q in range(0, i - 1):
            for r in range(j + 1, M):
                II += data[q][r]
        return II
    else:
        return 0


def calcIII(i, j, K, M, data):
    if i > 0 and j > 0:
        III = 0
        for q in range(0, i - 1):
            for r in range(0, j - 1):
                III += data[q][r]
        return III
    else:
        return 0


def calcD(i, j, K, M, data):
    return calcII(i, j, K, M, data) + calcIV(i, j, K, M, data)


def calcS(i, j, K, M, data):
    return calcI(i, j, K, M, data) + calcIII(i, j, K, M, data)


def calcVarG(data, Ps, Pd):
    K = data.shape[0]
    M = data.shape[1]
    VarG = 0
    for i in range(K):
        for j in range(M):
            VarG += data[i][j] * (Ps * calcD(i, j, K, M, data) - Pd * calcS(i, j, K, M, data)) ** 2
        VarG = VarG * (16 / (1000 * (Ps + Pd) ** 4))
    return VarG


def calcG(data, Ps, Pd):

    K = data.shape[0]
    M = data.shape[1]
    G = 0
    for i in range(K):
        for j in range(M):
            Ps += data[i][j] * calcI(i, j, K, M, data)
            Pd += data[i][j] * calcIV(i, j, K, M, data)
    Ps = Ps * 2
    Pd = Pd * 2

    G = (Ps - Pd) / (Ps + Pd)
    return G, Ps, Pd


def createFreq(data):
    K = data.shape[0]
    M = data.shape[1]
    weightColomn = np.zeros(M)
    weightRow = np.zeros(K)
    freq = np.zeros((K, M))
    for i in range(K):
        for j in range(M):
            weightRow[i] += data[i][j]
    for i in range(M):
        for j in range(K):
            weightColomn[i] += data[j][i]
    for i in range(K):
        for j in range(M):
            freq[i][j] = weightColomn[j] * weightRow[i] / 1000
    return freq

def finalCalculate(data, Ps, Pd):
    G, Ps, Pd = calcG(data, Ps, Pd)
    return abs(G)/calcVarG(data, Ps, Pd)

with open('names.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)

n = len(dataForAnalys)
p = len(dataForAnalys[0])

data = np.array(dataForAnalys)

X15 = data[:, 0]
X21 = data[:, 1]
X12 = data[:, 2]
X19 = data[:, 3]
X23 = data[:, 4]
X18 = data[:, 5]
X13 = data[:, 6]
X7 = data[:, 7]
X4 = data[:, 8]
X9 = data[:, 9]
Y = data[:, 10]

Y_X15 = pd.crosstab(Y, X15)
Y_X21 = pd.crosstab(Y, X21)
Y_X12 = pd.crosstab(Y, X12)
Y_X19 = pd.crosstab(Y, X19)
Y_X23 = pd.crosstab(Y, X23)
Y_X18 = pd.crosstab(Y, X18)
Y_X13 = pd.crosstab(Y, X13)

Y_X7 = pd.crosstab(Y, X7)
Y_X4 = pd.crosstab(Y, X4)
Y_X9 = pd.crosstab(Y, X9)
Ps = 0
Pd = 0

freq = createFreq(Y_X7.values)

print('\n')
G, Ps, Pd = calcG(freq, Ps, Pd)
print (G)
print(calcVarG(freq, Ps, Pd))
print(finalCalculate(freq, Ps, Pd))
