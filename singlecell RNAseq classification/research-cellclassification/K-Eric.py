from sklearn.datasets import load_iris
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_selection import VarianceThreshold #导入python的相关模块

# open up a datastore
store = pd.HDFStore('train_data.h5')

# Get the feature matrix (samples and their features)
feature_matrix_dataframe = store['rpkm']
name=feature_matrix_dataframe.index.values
gene=feature_matrix_dataframe.columns.values
data=feature_matrix_dataframe.values
sel=VarianceThreshold(threshold=5)  #表示剔除特征的方差大于阈值5的feature
new=sel.fit_transform(data)#返回的结果为选择的特征矩阵
new_feature_matrix_dataframe2=pd.DataFrame(new,index=name) #得到只剩下7916个feature的新数据

#注意，新数据的gene序列（columns）被抹掉了，但是这不重要

store.close()