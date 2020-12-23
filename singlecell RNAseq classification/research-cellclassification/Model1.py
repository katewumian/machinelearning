from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as ACCS
from sklearn import decomposition

import pandas as pd
clf = RandomForestClassifier(n_estimators = 100,max_depth=30,n_jobs=2,random_state=0)
store = pd.HDFStore('test_data.h5')
store1 = pd.HDFStore('train_data.h5')

feature_matrix_dataframe = store['rpkm']
feature_matrix_dataframe1 =  store1['rpkm']

labels_series = store['labels']
l = labels_series.values
labels_series1 = store1['labels']
l1= labels_series1.values
clf.fit(feature_matrix_dataframe1, l1)
pred_rfc = clf.predict(feature_matrix_dataframe)
#score_val = cross_val_score(clf,feature_matrix_dataframe1,l1,cv=5)
#print("5 fold CV")
#print(score_val)
score=ACCS(l, pred_rfc)
print("Acc")
print(score)

from sklearn.feature_selection import VarianceThreshold #导入python的相关模块
clf2 = RandomForestClassifier(n_estimators = 100,max_depth=15,n_jobs=5,random_state=0)
name=feature_matrix_dataframe1.index.values
gene=feature_matrix_dataframe1.columns.values
data=feature_matrix_dataframe1.values
sel=VarianceThreshold(threshold=5)  #表示剔除特征的方差大于阈值5的feature
new=sel.fit_transform(data)#返回的结果为选择的特征矩阵
#new_feature_matrix_dataframe=pd.DataFrame(new,index=name) #得到只剩下7916个feature的新数据

pca = decomposition.PCA(n_components=100)
pca.fit(new)
X = pca.transform(new)
new_feature_matrix_dataframe=pd.DataFrame(X, index=name)

store2 = pd.HDFStore('d:/CIS/all_data.h5')
feature_matrix_dataframe2 =  store2['rpkm']
name2=feature_matrix_dataframe2.index.values
gene2=feature_matrix_dataframe2.columns.values
data2=feature_matrix_dataframe2.values
sel2=VarianceThreshold(threshold=5)  #表示剔除特征的方差大于阈值5的feature
new2=sel2.fit_transform(data2)#返回的结果为选择的特征矩阵
#new_feature_matrix_dataframe2=pd.DataFrame(new2,index=name2) #得到只剩下7916个feature的新数据

pca2 = decomposition.PCA(n_components=100)
pca2.fit_transform()
pca2.fit(new2)
X2 = pca.transform(new2)
new_feature_matrix_dataframe2=pd.DataFrame(X2, index=name2)

from sklearn.model_selection import train_test_split
labels_series2 = store2['labels']
l2 = labels_series2.values
Xtrain,Xtest,Ytrain,Ytest = train_test_split(new_feature_matrix_dataframe2,l2,test_size=0.3,random_state=420)

clf2.fit(Xtrain, Ytrain)
pred_rfc2 = clf2.predict(Xtest)
#score_val = cross_val_score(clf,feature_matrix_dataframe1,l1,cv=5)
#print("5 fold CV")
#print(score_val)
score2=ACCS(Ytest, pred_rfc2)
print("Acc")
print(score2)

Xtrain1,Xtest1,Ytrain1,Ytest1 = train_test_split(feature_matrix_dataframe2,l2,test_size=0.3,random_state=420)