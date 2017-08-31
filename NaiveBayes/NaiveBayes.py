#coding=utf-8
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from MachineLearning.datasets.iris import load_data

class GaussianNB():
    def __init__(self):
        self.mu_sigma = None
        self.proba = None
    def fit(self,x,y):
        _y = pd.DataFrame(y,columns=['label'])
        label = pd.unique(_y['label'])
        label_proba= [_y[_y['label'] == i].count().values[0]/float(len(_y)) for i in label]
        label_proba_dict = dict(zip(label,label_proba)) #计算P(C)
        index = [_y[_y['label'] == i].index.values for i in label]
        label_index = dict(zip(label,index))
        _x = pd.DataFrame(x)
        model = []
        for c,index in label_index.iteritems():
            temp =  _x.loc[index]
            row= []
            row.append(c)
            for item in temp.columns:
                mu = temp.loc[:,item].mean()
                sigma = temp.loc[:,item].std()
                row.append(mu)
                row.append(sigma)
            model.append(row)
        self.mu_sigma = np.array(model)
        self.proba = label_proba_dict
    def __Gaussianproba(self,mu,sigma,x):
        import scipy.stats
        p = scipy.stats.norm(mu,sigma).pdf(x)
        return p


    def predict(self,data):
        results = []
        dim = len(data[0])
        for x in data:
            result = []
            for item in self.mu_sigma:
                k = 0
                P = 0
                temp = {}
                for i in xrange(1,2*dim+1,2):
                    mu = item[i]
                    sigma = item[i+1] 
                    p = self.__Gaussianproba(mu,sigma,x[k])
                    P += np.log(p)
                    k += 1
                temp[item[0]] = P*self.proba[item[0]]
                result.append(temp) 
            result.sort(key=lambda x:x.values(),reverse=True)
            results.append(int(result[0].keys()[0]))
        return results
        
            



if __name__ == '__main__':
    # clf = GaussianNB()
    # data = pd.read_csv('/home/coffee/study/MachineLearning/KNN/iris.data',header=None)
    # iris_types = data[4].unique()
    # for i, iris_type in enumerate(iris_types):
    #     data.set_value(data[4] == iris_type, 4, i)
    # x = data.iloc[:, :4]
    # y = data.iloc[:,-1].astype(np.int)
    # x = np.array(x)
    # y = np.array(y)
    # x, x_test, y, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    # clf.fit(x,y)
    # k = 0
    # for i,j in zip(y_test,clf.predict(x_test)):
    #     if i == j:
    #         k += 1
    # print 'accuracy:',float(k)/len(y_test)
    pass 