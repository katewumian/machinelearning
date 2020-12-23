#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as ACCS
import pandas as pd
clf = RandomForestClassifier(n_estimators = 100,max_depth=30,n_jobs=2,random_state=0)
store = pd.HDFStore('d:/CIS/test_data.h5')
store1 = pd.HDFStore('d:/CIS/train_data.h5')
feature_matrix_dataframe = store['rpkm']
feature_matrix_dataframe1 =  store1['rpkm']
labels_series = store['labels']
l = labels_series.values
labels_series1 = store1['labels']
l1= labels_series1.values


# In[3]:


clf.fit(feature_matrix_dataframe1, l1)


# In[4]:


pred_rfc = clf.predict(feature_matrix_dataframe)


# In[7]:


score=ACCS(l, pred_rfc)
print("before CV")
print(score)


# In[9]:


score_val = cross_val_score(clf,feature_matrix_dataframe1,l1,cv=5)
print("5 fold CV")
print(score_val)


# In[ ]:




