##author: Kate Mian Wu

import pandas as pd
import numpy as np
import re

csv_file = pd.read_csv('dump_S_1.csv', index_col=False).values
#print(csv_file[:10, :])
#print(csv_file.shape)  

unique, counts = np.unique(csv_file[:, 0].T, return_counts=True)
#print(type(unique))
#print(unique.shape) #(56,)
#print(unique)

csv_file2 = pd.read_csv('cell_type.csv', index_col=False).values

#print(csv_file2.shape) #(46,1)
#print(csv_file2)

ss = []

for item in unique:
    if item not in csv_file2[:,0]:
        print(item)
    else:
        ss.append(item)

similarity_matrix = []
[rows, columns] = csv_file.shape
for i in range(rows):
    if csv_file[i, 0] in ss:
        similarity_matrix.append(csv_file[i, :])
##print(similarity_matrix)

df = pd.DataFrame(similarity_matrix)
##df.to_csv('similarity_matrix.csv',index=False, header=False)

simi_array = df.values
[rs, cs] = simi_array.shape
simi10 = []
##re.findall(r"\d+\.?\d*", simi_array[i, 2]
for i in range(rs):
    # print(type(simi_array[i, 2]))
    if len(re.findall('human', simi_array[i, 2])) != 0:
        continue
    else:
        if len(re.findall('CD34-positive', simi_array[i, 2])) != 0:
            continue
        else:
            if eval(simi_array[i, 2]) > 0.1:
                simi10.append(simi_array[i, :])
                # print(simi_array[i, :])

df1 = pd.DataFrame(simi10)
df1.to_csv('simi10.csv', index = False, header = False)
print(len(simi10))
unique1, counts1 = np.unique(df.values[:, 0].T, return_counts=True)
##print(type(unique1))

unique1.sort
print(unique1.shape) #(44,)
print(unique1)
df2 = pd.DataFrame(unique1)
##df2.to_csv('cell_similarity_sort.csv', index = False, header = False)

##for item in csv_file2[:, 0]:
##    if item == 'CL:0000353 blastoderm cell':
##        print(item)
##    if item not in unique1:
##        print(item)
##print('CL:0000353 blastoderm cell' in unique1)
