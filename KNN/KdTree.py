#coding=utf-8
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import math


class KdNode(object):
    def __init__(self,point,label,split,left,right):
        self.point = point #此结点的点
        self.label = label
        self.split = split #划分维度
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self,data,labels):
        k = len(data[0])
        def CreateNode(split,dataset,labels):
            if len(dataset)==0:
                return None
            data = np.column_stack((dataset,labels))
            data = list(data)
            data.sort(key=lambda x:x[split])
            data = np.array(data)
            dataset = data[:,:-1]
            labels = data[:,-1]
            split_pos = len(dataset)//2
            median = dataset[split_pos]
            label = labels[split_pos]
            split_next = (split+1)%k
            return KdNode(median,label,split,CreateNode(split_next,dataset[:split_pos],labels[:split_pos]),CreateNode(split_next,dataset[split_pos+1:],labels[split_pos+1:]))

        self.root = CreateNode(0,data,labels)

def preorder(root):
    print root.point
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)



def find_nearest(tree,point):
    nearest_p = tree.point
    min_dist = ComputeDistance(nearest_p,point)
    search_path = []
    temp_node = tree
    nearest_list = []
    
    while temp_node:
        search_path.append(temp_node)
        dist = ComputeDistance(temp_node.point,point)
        nearest = {'point':None,'dist':None,'label':None}
        nearest['dist'] = dist
        nearest['point'] = temp_node.point
        nearest['label'] = temp_node.label
        nearest_list.append(nearest)
        if min_dist>dist:
            min_dist = dist
            nearest_p = temp_node.point
        d = temp_node.split
        if point[d] <= temp_node.point[d]:
            temp_node = temp_node.left
        else:
            temp_node = temp_node.right
    
    while search_path:
        nearest = {'point':None,'dist':None,'label':None}
        cur_point = search_path.pop()
        d = cur_point.split
        if abs(point[d]-cur_point.point[d])<min_dist:
            if point[d] <= cur_point.point[d]:
                temp_node = cur_point.right
            else:
                temp_node = cur_point.left
            if temp_node:
                search_path.append(temp_node)
                dist = ComputeDistance(point,temp_node.point)
                if min_dist > dist:
                    min_dist = dist
                    nearest_p = temp_node.point 
                    nearest['dist'] = min_dist
                    nearest['point'] = nearest_p
                    nearest['label'] = temp_node.label
                    nearest_list.append(nearest)

    return nearest_list

def ComputeDistance(p1,p2):
    sum = 0.0  
    for i in range(len(p1)):  
        sum = sum + (p1[i] - p2[i]) * (p1[i] - p2[i])  
    return math.sqrt(sum)  


def KNNClassifier(inputdata,datasets,labels,k):
    kd = KdTree(list(datasets),list(labels))
    nearest_list = find_nearest(kd.root,inputdata)
    nearest_list.sort(key=lambda x:(x.get('dist',0)))
    result ={}
    for i in xrange(k):
        result[nearest_list[i].get('label',0)] = nearest_list[i].get('label',0)+1
    result = sorted(result.iteritems(),key = lambda x:x[1],reverse=True)
    return result[0][0]

if __name__ == '__main__':
    import time
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