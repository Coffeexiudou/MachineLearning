#coding=utf-8
import pandas as pd 
import numpy as np 
from scipy.stats import entropy

class DecisionTree:
    def __init__(self):
        self.tree = None 

    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            pass
        else:
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError('pandas.DataFrame required for X')
        if isinstance(y,pd.Series):
            pass
        else:
            try:
                y = pd.Series(y)
            except:
                raise TypeError('pandas.Series required for y') 
        featureIndexList = list(X.columns.values)
        self.tree = self.__create_tree(X,y,featureIndexList)
        return self.tree

    def predict(self,X):
        if self.tree == None:
            raise ValueError('Please fit first')
        if isinstance(X,pd.DataFrame):
            pass
        else:
            try:
                X = pd.DataFrame(X)
            except:
                raise TypeError('pandas.DataFrame required for X')
        def __classify(tree,sample):
            firstFeatureIndex = tree.keys()[0]
            firstFeatureDict = tree[firstFeatureIndex]
            featureVal = sample[firstFeatureIndex]
            value = firstFeatureDict[featureVal]
            if isinstance(value,dict):
                label = __classify(value,sample)
            else:
                label = value
            return label
        results = []
        for i in xrange(X.shape[0]):
            results.append(__classify(self.tree,X.loc[i]))
        return results

                
    def __compute_entropy(self,p):
        """
        计算信息熵，底数为e
        """
        ent = 0.
        for i in p:
            ent-=i*np.log2(i)
        return ent 
       # return entropy(p) 
    
    def __compute_majority_cnt(self,y):
        """
        计算叶节点中最多的种类
        """
        return y.value_counts().index[0]

    def __compute_bestsplit(self,X,y,featureIndexList):
        pY = y.value_counts().values/float(len(y))
        oldEnt = self.__compute_entropy(pY)
        gain = {}
        index = {}
        for featureIndex in featureIndexList:
            value = X.loc[:,featureIndex].value_counts().index.values #特征featureIndex的不同种类
            count = X.loc[:,featureIndex].value_counts().values #特征featureIndex的每类数目
            newEntList = []
            index[featureIndex] = []
            for val,cnt in zip(value,count):
                indexDict = {}
                sampleIndex = X[X.loc[:,featureIndex] == val].index.values #特征featureIndex中类别为val的样本索引
                indexDict[val] = sampleIndex
                index[featureIndex].append(indexDict)
                p = y[sampleIndex].value_counts().values/float(cnt) #特征featureIndex中类别为val的样本类别分布概率
                entDi = self.__compute_entropy(p) 
                newEntList.append(entDi)
            newEntList = np.array(newEntList)
            pDi = count/float(X.shape[0]) #特征featureIndex中每个val对应的样本数量比例
            newEnt = np.sum(newEntList*pDi)
            gain[featureIndex] = oldEnt - newEnt
        gainList = sorted(gain.iteritems(),key = lambda d:d[1],reverse=True) #根据信息增益值排序，取最大
        bestSplitFeatureIndex = gainList[0][0]
        return  index[bestSplitFeatureIndex],bestSplitFeatureIndex
        
    def __create_tree(self,X,y,featureIndexList):
        if len(featureIndexList) == 0:
        #当属性集为空时，返回当前结点中样本最多的类别
            return self.__compute_majority_cnt(y)
        listY = list(y)
        if listY.count(listY[0]) == len(listY):
        #所有样本类别都相同时，返回该类别
            return listY[0]
        sampleIndexByBestFeature,bestSplitFeatureIndex = self.__compute_bestsplit(X,y,featureIndexList)
        featureIndexList.remove(bestSplitFeatureIndex)
        mytree = {bestSplitFeatureIndex:{}}
        for featureDict in sampleIndexByBestFeature:
            featureVal = featureDict.keys()[0]
            featureValIndex = featureDict.values()[0]
            mytree[bestSplitFeatureIndex][featureVal] = self.__create_tree(X.ix[featureValIndex],y[featureValIndex],featureIndexList)
        return mytree
        


        


    

if __name__ == '__main__':
    a = DecisionTree()
    x = np.array([[2,1,2,1,1,1],[2,1,1,1,1,1],[1,1,2,1,1,1],[3,1,1,1,1,1],[1,2,1,1,2,2],[2,2,1,2,2,2],[2,2,1,1,2,1],[2,2,2,2,2,1],[1,3,3,1,3,2],[3,3,3,3,3,1],[3,1,1,3,3,2],[1,2,1,2,1,1],[3,2,2,2,1,1],[2,2,1,1,2,2],[3,1,1,3,3,1],[1,1,2,2,2,1]])
    y = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    x = pd.DataFrame(x,columns=['seze','gendi','qiaosheng','wenli','qibu','chugan'])
    y = pd.Series(y)
    tree=a.fit(x,y)
    test = np.array([[1,1,1,1,1,1]])
    test = pd.DataFrame(test,columns=['seze','gendi','qiaosheng','wenli','qibu','chugan'])
    print a.predict(test)