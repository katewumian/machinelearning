from sklearn.datasets import load_iris
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_selection import VarianceThreshold #导入python的相关模块

# open up a datastore
store = pd.HDFStore('train_data.h5')
store1 = pd.HDFStore('test_data.h5')
# Get the feature matrix (samples and their features)
feature_matrix_dataframe = store['rpkm']     #train的数据框架
feature_matrix_dataframe_test = store1['rpkm']   #test的数据框架

all_in = pd.concat([feature_matrix_dataframe,feature_matrix_dataframe_test],axis=0,join='inner')

name=all_in.index.values   #提取行标签

data=all_in.values  #提取数据丢掉标签（总和数据）
sel=VarianceThreshold(threshold=15)  #表示剔除特征的方差大于阈值15的feature
new=sel.fit_transform(data)#返回的结果为选择的特征矩阵
new_all=pd.DataFrame(new,index=name) #得到只剩下feature的新数据
a=new_all.iloc[:5,3]
new_train = new_all.iloc[:21389,]  #新的train数据 
new_test = new_all.iloc[21389: ,]  #新的test数据

#注意，新数据的gene序列（columns）被抹掉了，但是这不重要
#唯一的问题是所有的数据一起提取方差了