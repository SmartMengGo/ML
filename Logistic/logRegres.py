import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataMat=[]
    label=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        label.append(lineArr[2])
    return dataMat,label

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatin,classLabels):
    dataMat=np.array(dataMatin).astype(np.float)
    labelMat=np.array(classLabels).astype(np.float)
    m,n=dataMat.shape
    labelMat=labelMat.reshape((m,1))
    alpha=0.001
    maxCyc=500
    weights=np.ones((n,1))
    for k in range(maxCyc):
        h=sigmoid(np.dot(dataMat,weights))
        err=np.subtract(labelMat,h)
        weights+=(np.dot(dataMat.T,err))*alpha
    return weights
    


    
def plotFit(weights):
    dataMat,labels=loadData()
    dataArr=np.array(dataMat)
    m=np.shape(dataArr)[0]
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for i in range(m):
        if int(labels[i])==1:
            x1.append(dataArr[i,1])
            y1.append(dataArr[i,2])
        if int(labels[i])==0:
            x2.append(dataArr[i,1])
            y2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x1,y1,s=30,c='red',marker='s')
    ax.scatter(x2,y2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(np.dot(inX,weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(trainingSet, trainingLabels)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate    



colicTest()
