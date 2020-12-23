import pandas as pd

store = pd.HDFStore('all_data.h5')

labels_series = store['labels']

labels_series.columns = ['Id', 'Labels']
labels_series.dropna(how="all", inplace=True) # drops the empty line at file-end

labels_series.tail()

#In order to write data to disks, you can change the file name to all, train, test
df.to_csv('train_cell_type_sort.csv',index=False, header=False)

store.close()


