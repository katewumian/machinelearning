import matplotlib.pyplot as plt
import numpy as np

def plotMLE(X,theta):
    x = np.linspace(0,1,num=101)
    y = sum(X)*np.log(1-x)+len(X)*np.log(x)
    theta_c = np.array(theta)
    y_c = sum(X)*np.log(1-theta_c)+len(X)*np.log(theta_c)
    max_index = np.argmax(y_c)

    plt.plot(x,y)
    plt.scatter(theta_c, y_c, label='candidate',s=10)
    plt.scatter(theta_c[max_index], y_c[max_index], label='MLE',s=20)
    plt.axhline(y=y_c[max_index],c="#4d4d4d", ls="--", lw=1)
    plt.axvline(x=theta_c[max_index], c="#4d4d4d", ls="--", lw=1)
    plt.annotate("theta:"+str('{:.2f}'.format(theta_c[max_index])),(theta_c[max_index], y_c[max_index]),xytext=(theta_c[max_index], y_c[max_index]*2),arrowprops=dict(arrowstyle='->'))
    plt.legend()
    plt.title("MLE")  # 图形标题
    plt.xlabel("theta value")  # x轴名称
    plt.ylabel("log of likelihood")  # y 轴名称
    plt.show()

if __name__ == "__main__":
    plotMLE([0, 1, 2, 3, 4, 5],[0.1, 0.2, 0.3,0.4])
    theta = np.linspace(0.01,0.99,99)
    file_name = './hw1_dataset.txt'  # 定义数据文件
    data = np.loadtxt(file_name, dtype='int32', delimiter='\n')  # 获取数据
    plotMLE(data[:1000], theta)
    plotMLE(data[:10000], theta)
