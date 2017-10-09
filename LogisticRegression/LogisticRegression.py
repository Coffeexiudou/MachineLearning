#coding=utf-8
import numpy as np 

class LogisticRegression:
    def __init__(self):
        self.weigh = None 

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def __gradAscend(self,x,y,learning_rate,penalty,lambda_,max_iter,optimizers):
        m,n = np.shape(x)
        b = np.ones(m)  
        x = np.mat(np.column_stack((b,x))) #添加偏置b　m*(n+1)
        weigh = np.mat(np.ones((n+1,1))) #权值矩阵 (n+1)*1  包括w0
        if penalty == None:
            if optimizers == 'BGD':
                for i in xrange(max_iter):
                    error = y-self.sigmoid(x*weigh)
                    weigh = weigh + learning_rate*x.T*error  #梯度上升 似然函数求最大值
                return weigh
            elif optimizers == 'SGD':
                pass
            else:
                raise NameError('Just support SGD and BGD')
        elif penalty == 'l2':
            if optimizers == 'BGD':
                for i in xrange(max_iter):
                    error = y-self.sigmoid(x*weigh)
                    weigh = weigh + learning_rate*(x.T*error + lambda_*weigh/m)
                return weigh

        

    def fit(self,x,y,learning_rate=0.01,penalty=None,lambda_=None,max_iter=100,optimizers='BGD'):
        '''
        parameters:
        x:train_data shape:m*n
        y:train_label shape:m*1
        learning_rate:学习率
        penalty:正则项　只能l2
        lambda_:正则项参数
        max_iter:最大迭代次数
        optimizers:优化器 SGD,BGD
        '''
        self.weigh = self.__gradAscend(x,y,learning_rate,penalty,lambda_,max_iter,optimizers)

    def predict(self,x):
        m,n = x.shape
        b = np.ones(m) 
        x = np.mat(np.column_stack((b,x)))
        result = self.sigmoid(x*self.weigh)
        length = len(result)
        for i in xrange(length):
            if result[i]>0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result

def loadData():  
    train_x = []  
    train_y = []  
    fileIn = open('/home/coffee/study/MachineLearning/LogisticRegression/test.txt')  
    for line in fileIn.readlines():  
        lineArr = line.strip().split()  
        train_x.append([float(lineArr[0]), float(lineArr[1])])  
        train_y.append(float(lineArr[2]))  
    return np.array(train_x), np.mat(train_y).T
if __name__ == '__main__':
    x,y=loadData()
    x_train = x[:50]
    y_train = y[:50]
    x_test = x[50:]
    y_test = y[50:]
    model = LogisticRegression()
    model.fit(x_train,y_train,learning_rate=0.01,max_iter=200,penalty='l2',lambda_=2)
    print model.weigh
    result = model.predict(x_test)
    l = len(result)
    k = 0
    for i in range(l):
        if result[i] == y_test[i]:
            k+=1
    print float(k)/l
