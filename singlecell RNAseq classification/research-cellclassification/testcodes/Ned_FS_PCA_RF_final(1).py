import seaborn as sns
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.feature_selection import VarianceThreshold  # 导入python的相关模块
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as ACCS
import pandas as pd
from sklearn import tree
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# open up a datastore
store1 = pd.HDFStore('C:/ftp/ml_10701_ps5_data.tar/test_data.h5')
store = pd.HDFStore('C:/ftp/ml_10701_ps5_data.tar/train_data.h5')

print (store)
print (store.keys())
print (store.info())

# Get the feature matrix (samples and their features)
labels_series_train = store['labels']
feature_matrix_dataframe_train = store['rpkm']
feature_matrix_dataframe_test = store1['rpkm']  # test的数据框架

print (type(labels_series_train))

print ('labels should be 21389,1')
print (labels_series_train.shape)
       
print (type(feature_matrix_dataframe_test))
print ('train data feature shape should be 21389,20499：')
print (feature_matrix_dataframe_train.shape)

feature_matrix_dataframe_train['labels'] = labels_series_train #合并原来的train label series和rpkm dataframe

print (type(feature_matrix_dataframe_train))
print ('after addition should be 21389,20500:')
print (feature_matrix_dataframe_train.shape)
print (feature_matrix_dataframe_train.labels)

Test_cell_types = [
         'CL:0000353 blastoderm cell',
         'CL:0002322 embryonic stem cell',
         'UBERON:0002107 liver',
         'UBERON:0001851 cortex',
         'UBERON:0000115 lung epithelium',
         'UBERON:0000922 embryo',
         'CL:0000746 cardiac muscle cell',
         "UBERON:0001954 Ammon's horn",
         'CL:0000037 hematopoietic stem cell',
         'UBERON:0000044 dorsal root ganglion',
         'CL:0002321 embryonic cell',
         'UBERON:0001003 skin epidermis',
         'CL:0002319 neural cell',
         'UBERON:0002048 lung',
         'CL:0000137 osteocyte',
         'UBERON:0001898 hypothalamus',
         'CL:0000540 neuron',
         'UBERON:0001264 pancreas',
         'CL:0000235 macrophage',
         'UBERON:0000955 brain',
         'UBERON:0000966 retina']

#进行筛选
feature_matrix_dataframe = feature_matrix_dataframe_train[feature_matrix_dataframe_train.labels.isin(Test_cell_types)]

#测试是否完成筛选
print (feature_matrix_dataframe.loc[feature_matrix_dataframe['labels'] == 'UBERON:0000966 retina'])

print ('shape after Test cell filter: ')
print (feature_matrix_dataframe.shape)

labels_series = feature_matrix_dataframe['labels']
train_data_lab = labels_series.values      #selected train data label

#去除label
feature_matrix_dataframe.drop(columns=['labels'],inplace = True)  # train的数据框架

print ("shape after drop the labels:")
print (feature_matrix_dataframe.shape)

all_in = pd.concat([feature_matrix_dataframe,feature_matrix_dataframe_test],axis=0,join='inner')

print (all_in.shape)

name=all_in.index.values   #提取行标签

data=all_in.values  #提取数据丢掉标签（总和数据）

sel=VarianceThreshold(threshold=15)  #表示剔除特征的方差小于阈值 i的feature
new=sel.fit_transform(data)#返回的结果为选择的特征矩阵
new_all=pd.DataFrame(new,index=name) #得到只剩下feature的新数据
new_train = new_all.iloc[:10944,]  #新的train数据 
new_test = new_all.iloc[10944: ,]  #新的test数据

    #注意，新数据的gene序列（columns）被抹掉了，但是这不重要
    #唯一的问题是所有的数据一起提取方差了

pca = decomposition.PCA(n_components = 50)  #这边可以用MLE吗？
new_train_afterPCA = pca.fit_transform(new_train.values)
new_test_afterPCA = pca.transform(new_test.values)
            #降维
new_train_afterPCA_da = pd.DataFrame(new_train_afterPCA, index=new_train.index.values)
new_test_afterPCA_da = pd.DataFrame(new_test_afterPCA, index=new_test.index.values)
print(new_train_afterPCA_da.shape[0])
print(new_train_afterPCA_da.shape[1])

labels_series1 = store1['labels']
test_data_lab = labels_series1.values         #test data label

clf = RandomForestClassifier(min_samples_leaf=55, n_estimators=150, max_depth=30, n_jobs=2, random_state=0)
clf.fit(new_train_afterPCA_da, train_data_lab)
# print(new_test_afterPCA_da)
# print(type(new_test_afterPCA_da))
# print(new_test_afterPCA_da.shape)

pred_rfc = clf.predict(new_test_afterPCA_da)

# print(pred_rfc)
# print(type(pred_rfc))
# print(pred_rfc.shape)

score=ACCS(test_data_lab, pred_rfc)
print("Accuracy")
print(score)