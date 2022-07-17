import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
	 
point_data=[] #무작위 점 분포
for p in range(1):
  for q in range(0,11):
    a=random.randint(78,82)
    point_data.append([q,a+2*q])
  print(point_data)
  
x=[i[0] for i in point_data] 
y=[i[1] for i in point_data] 
x_r = np.array(x) 
y_r= np.array(y) 
#선언(기울기,절편,학습률,반복수,c-r그래프 점,c-w,b그래프 점)
W = -40
b = 60
rate = 0.001
repeat= 10001 
moving_point=[]
 #cost(a,b)-point_data에 따른
def f(a,b):
  return (1/len(x_r))*((sum(x_r)*a+len(x_r)*b-sum(y_r))**2)
	 
#경사하강
for i in range(repeat):
  y_p = W * x_r + b 
  mis = y_r-y_p
  cost=(1/len(x_r))*sum((mis)**2)
  W_gradient = -(1 / len(x_r)) * sum(x_r * (mis)) 
  b_gradient = -(1 / len(x_r)) * sum(mis) 
  W = W - rate * W_gradient
  b = b - rate * b_gradient
  if i%1000==0: 
    print("반복:",i, "\t기울기:",round(W,2), "\t절편:",round(b,2),"\t비용:",round(cost,2))
  if i%1==0:
    moving_point.append([W,b])
print(W,"x+",b)
#x-y(무작위점)
fig = plt.figure(figsize=(4,5))
plt.figure(1)
x0=x_r
y0=y_r
plt.scatter(x0, y0,c='b') 
#x-y(선형회귀)
x=np.arange(-1,11,1)
y=W*x+b
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Liner Regression")

#cost-wb그래프(경사)
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection="3d")
plt.figure(2)
a = np.arange(-50, 50, 0.25)
b = np.arange(0, 150, 0.25)
a, b = np.meshgrid(a, b)
c = f(a,b)+1000
surf = ax.plot_surface(a, b, c,cmap='rainbow', alpha=0.7)

#cost-wb그래프(점)
x=[i[0] for i in moving_point]
y=[i[1] for i in moving_point]
x1=np.array(x)
y1=np.array(y)
z1=f(x1,y1)
ax.scatter(x1,y1,z1,c='r',s=10)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('cost')
plt.title(a)
plt.title("cost(a,b) Graph")


plt.show()

