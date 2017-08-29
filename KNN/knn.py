#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time 


def computeDistance(inputdata,datasets):
    num = datasets.shape[0]
    diff = np.tile(inputdata,(num,1)) - datasets
    square = diff ** 2
    square_sum = np.sum(square,axis=1)
    distabce = square_sum ** 0.5
    return distabce

def KNNClassifier(inputdata,datasets,labels,k):
    distance = computeDistance(inputdata,datasets)
    dis_index = np.argsort(distance)
    result_dict = {}
    for i in xrange(k):
        label = labels[dis_index[i]]
        result_dict[label] = result_dict.get(label,0) + 1
    result = sorted(result_dict.iteritems(),key = lambda x:x[1],reverse=True)
    return result[0][0]



if __name__ == '__main__':
    start = time.clock()
    data = pd.read_csv('iris.data',header=None)
    iris_types = data[4].unique()
    for i, iris_type in enumerate(iris_types):
        data.set_value(data[4] == iris_type, 4, i)
    x = data.iloc[:, :4]
    y = data.iloc[:,-1].astype(np.int)
    x = np.array(x)
    y = np.array(y)
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    print x.shape[1]
    times = 0
    for i,value in enumerate(x_test):
        predict = KNNClassifier(value,x,y,5)
        print 'knn predict:',predict
        print 'true:',y_test[i]
        if predict == y_test[i]:
            times += 1
    print 'accuracy:',float(times)/len(y_test)
    end = time.clock()
    print 'time:',end-start