#coding=utf-8
import pandas as pd 
import numpy as np
def load_data():
    data = pd.read_csv('/home/coffee/study/MachineLearning/datasets/iris.data',header=None)
    iris_types = data[4].unique()
    for i, iris_type in enumerate(iris_types):
        data.set_value(data[4] == iris_type, 4, i)
    x = data.iloc[:, :4]
    y = data.iloc[:,-1].astype(np.int)
    x = np.array(x)
    y = np.array(y)
    return x,y