# My solution for https://www.hackerrank.com/challenges/predicting-house-prices/problem

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


parseInput()
