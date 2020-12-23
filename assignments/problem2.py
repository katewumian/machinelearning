import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def plotMLE(X=[0],theta=[0]):
    n=len(X)
    i=0
    sum=0
    while i<n:
        sum+=X[i]
        i+=1
    if sum==0:
        return
    average=sum/n
    realtheta=1/(average+1)   #realtheta是最大似然估计值
    x=np.linspace(0,1,100)
    y=sum*np.log(1-x)+n*np.log(x)
    plt.plot(x,y)
    plt.show()  #以上画图，以下求theta
    i=1
    length=len(theta)
    absvalue=abs(realtheta-theta[0])  #absvalue是和realtheta差的绝对值
    candidate=theta[0] 
    while i<length:
        if abs(theta[i]-realtheta)<absvalue:
            absvalue=abs(theta[i]-realtheta)
            candidate=theta[i]
        i+=1
    return candidate


#以下为第二题
theta=np.arange(0.01,1,0.01) #构造theta数组
file_name = 'C:/Users/brigh/Desktop/CIS课程作业/programme2/hw1_dataset.txt'  # 定义数据文件
data = np.loadtxt(file_name, dtype='int32', delimiter='\n')  # 获取数据
print(plotMLE(data[:1000],theta))
print(plotMLE(data[:10000],theta))
print(plotMLE(data,theta))