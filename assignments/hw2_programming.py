import matplotlib.pyplot as plt
import numpy as np

def plotMLE(X, theta):
    x = np.linspace(0, 1, num = 1001)#x随机变量，定义横坐标的每一个点。linspace是Matlab中的均分计算指令，用于产生x1,x2之间的N点行线性的矢量。精度为0.001
    y = sum(X) * np.log(1 - x) + len(X) * np.log(x)#定义纵坐标的值的计算函数

    theta_c = np.array(theta)#把送入的列表参数theta，转化为array类型
    y_c = sum(X) * np.log(1 - theta_c) + len(X) * np.log(theta_c)
    max_index = np.argmax(y_c)#y_c是一个array，从这个array中找到最大值，返回最大值对应的index

    plt.plot(x, y)#画图用matplotlib.pyplot  ???具体是什么功能

    plt.scatter(theta_c, y_c, label = 'candidate', s = 10)#画圆点，散点
    plt.scatter(theta_c[max_index], y_c[max_index], label = 'MLE', s = 20)#画另一种圆点，散点，半径更大一些

    plt.axhline(y = y_c[max_index], c = "#4d4d4d", ls = "--", lw = 1)#画横线
    plt.axvline(x = theta_c[max_index], c = "#4d4d4d", ls = "--", lw = 1)#画竖线
    plt.annotate("theta:" + str('{:.2f}'.format(theta_c[max_index])), (theta_c[max_index], y_c[max_index]), xytext = (theta_c[max_index], y_c[max_index] * 2), arrowprops = dict(arrowstyle = '->'))
    plt.legend()
    plt.title("MLE")#画出图形标题
    plt.xlabel("theta value")#画出x轴名称
    plt.ylabel("log of likelihood")#画出y轴名称
    plt.show()#展示出来

if __name__ == "__main__":
    #plotMLE([0, 1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
    theta = np.linspace(0.01, 0.99, 100)
    file_name = './hw1_dataset.txt'
    data = np.loadtxt(file_name, dtype = 'int32', delimiter = '/n')
    plotMLE(data[:1000], theta)
    
