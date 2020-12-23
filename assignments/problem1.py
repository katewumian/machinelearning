import numpy as np
import matplotlib.pyplot as plt
import math
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

X=[0,1,2,3,4,5]
theta=[0.1,0.282,0.3]
print(plotMLE(X,theta))