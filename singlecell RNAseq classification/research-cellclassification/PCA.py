import pandas as pd
import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

store = pd.HDFStore('all_data.h5')

feature_matrix_dataframe = store['rpkm']
X = feature_matrix_dataframe.values
# print(type(X), X.shape)

mu = np.mean(X, axis=0)
Z = X - mu
C = np.cov(Z, rowvar=False)

[lam, V] = LA.eigh(C)
lam = np.flipud(lam)
V = np.flipud(V.T)

P = np.dot(Z, V.T)

R = np.dot(P, V)
# print(np.allclose(Z, R)) #true



# Xrecover = R + mu
# Xrec1000 = (np.dot(P[:, 0:1000], V[0:1000, :])) + mu

#降到2维的可视化
#np.random.seed(0)
#randomorder = np.random.permutation(np.arange(len(X)))


# Set colors
#opacity = 0.25
#cols = np.zeros((len(X), 4))
#cols[0] = [1, 0, 0, opacity]
#cols[1] = [0, 1, 0, opacity]
#设置不同类的颜色

# Draw scatter plot
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, facecolor='black')
#ax.scatter(P[randomorder, 1], P[randomorder, 0], s=50, linewidths=0, facecolors=cols[randomorder, :], marker="o")
#ax.set_aspect('equal')

#plt.gca().invert_yaxis()
#plt.show()

store.close()
