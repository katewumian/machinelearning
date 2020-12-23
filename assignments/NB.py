import pandas as pd
import math
file_name = './adult.txt'  # 定义数据文件
df =  pd.read_csv(file_name,sep=',',names=['1', '2','3','4', '5','6','7', '8','9','10', '11','12','13', '14','label'])
a = df.loc[:,'label'].value_counts()
prior_morethan50k = '{:.4f}'.format(a[0]/df.shape[0])
prior_lessthan50k = '{:.4f}'.format(a[1]/df.shape[0])

#discrete 2 4 6 7 8 9 10 15
b = df.label
class1 = df[df['label'].str.contains('>50K')]
class2 = df[df['label'].str.contains('<=50K')]

c = class1.loc[:,'2'].value_counts()
d1 = class1.loc[:,'1'].mean()
d2 = class1.loc[:,'1'].var()

c1 = class2.loc[:,'2'].value_counts()
d11 = class2.loc[:,'1'].mean()
d21 = class2.loc[:,'1'].var()
#这里只做了两个属性，一个连续，一个离散
#15个属性的话可以考虑用np来做，原理都是一样的

testfile_name = './adulttest.txt'  # 定义数据文件
df1 =  pd.read_csv(file_name,sep=',',names=['1', '2','3','4', '5','6','7', '8','9','10', '11','12','13', '14','label'])
#这里只算了第一条测试集的数据，不过前十条都是一样的
print()

class1_age_prob = (a[0]/df.shape[0]) *(math.exp(-0.5 * ((df1.iloc[0,0] - d1) / d2)) / (math.sqrt(2*math.pi*d2)))
class2_age_prob = (a[1]/df.shape[0]) *(math.exp(-0.5 * ((df1.iloc[0,0] - d11) / d21)) / (math.sqrt(2*math.pi*d21)))
class1_workclass_prob = c[df1.iloc[0,1]]/class1.shape[0]
class2_workclass_prob = c1[df1.iloc[0,1]]/class2.shape[0]

class1_post_prob = math.log(class1_age_prob*class1_workclass_prob)
class2_post_prob = math.log(class2_age_prob*class2_workclass_prob)
print("prior(more than 50k) is "+str(prior_morethan50k))
print("prior(less than 50k) is "+str(prior_lessthan50k))
print("Class>50K:")
print("age: mean ="+'{:.4f}'.format(d1)+", var = "+'{:.4f}'.format(d2))
print()
for i in range(c.shape[0]):
    print("workclass: "+c.index[i-1]+"="+'{:.4f}'.format(c[i-1]/class1.shape[0]))
print()
print("Class<=50K:")
print("age: mean ="+'{:.4f}'.format(d11)+", var = "+'{:.4f}'.format(d21))
print()
for i in range(c1.shape[0]):
    print("workclass: "+c1.index[i-1]+"="+'{:.4f}'.format(c1[i-1]/class2.shape[0]))

#根据前两个属性算出两个class的后验概率，选择最大的即可，在本次情况中判断正确，为>50K
print(class1_post_prob)
print(class2_post_prob)
