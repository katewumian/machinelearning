import pandas as pd

# open up a datastore

#all, train, test
store = pd.HDFStore('test_data.h5')

labels_series = store['labels']
print(type(labels_series))
print(labels_series.shape)

ss = []
for item in labels_series:
    if item not in ss:
        ss.append(item)
print(len(ss))
ss.sort()

print(ss)
df = pd.DataFrame(ss)

#In order to write data to disks, you can change the file name to all, train, test
df.to_csv('train_cell_type_sort.csv',index=False, header=False)

store.close()
