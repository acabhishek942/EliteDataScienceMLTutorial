# My solution for https://www.hackerrank.com/challenges/predicting-house-prices/problem
# this program gets a score of (9.82 / 10)
import numpy as np

def parseInput():
    # put the number of features in the 'feature' variable
    # put the number of training examples in the 'number' variable
    features, number = list(map(int,input().split()))


    # trainData : list of list containing all the training exapmples features
    # trainLabels : prices for the training examples in the same sequence as train_data
    trainData = []
    trainLabels = []
    for i in range(number):
        trainData.append(list(map(float,input().split())))
    for i in trainData:
        trainLabels.append(i[-1])
        i.pop()
    # testNumber : Number of test set examples
    # testData : list of list containing the test examples
    testNumber = int(input())
    testData = []
    for i in range(testNumber):
        testData.append(list(map(float,input().split())))

    return (trainData, trainLabels, testData)

def changeToMatrix(trainData, trainLabels, testData):
    trainDataMatrix = np.array(trainData)
    trainLabelsMatrix = np.array(trainLabels)
    trainDataMatrixTranspose = np.transpose(trainDataMatrix)
    xtransposexX = np.matmul(trainDataMatrixTranspose, trainDataMatrix)
    xtransposexXInverse = np.linalg.pinv(xtransposexX)
    xtransposexXInverseintoX = np.matmul(
                            xtransposexXInverse, trainDataMatrixTranspose)
    thetaMatrix = np.matmul(xtransposexXInverseintoX, trainLabelsMatrix)
    return thetaMatrix
trainData, trainLabels, testData = parseInput()


thetaMatrix = changeToMatrix(trainData, trainLabels, testData)
for j in testData:
    prediction  = 0
    for i, k  in zip(j, thetaMatrix):
        prediction += i * k
    print ("%2.f" % prediction)
