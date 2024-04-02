from collections import deque
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import islice
num = 1500
offset = 10
def f(x):
    return math.log(1/5*x+offset) - math.log(offset)

def f1(x):
    return 10 -  1000/(1/8*(x+250)+100) + 1000/(1/8*250+100) - 10

def f2(x):
    # return 10 -  1000/(1/5*(x+250)+100) + 1000/(150) - 10
    return 2 / (1 + np.exp(- x * 0.02)) - 1

# plt.plot([x for x in range(num)], [f1(x) for x in range(num)], label="f1")
print(np.random.randint(0, 10, 1))
plt.plot([x for x in range(num)], [f2(x) for x in range(num)], label="f2")
# plt.plot([x for x in range(num)], [f(x) for x in range(num)], label="f")
plt.axvline(200)
plt.legend()
plt.show()

# A = np.array([1, 3, 5, 7, 9, 11])
# where1 = np.where(A == 1)
# print("x%d"%(where1[0]))

# a = np.array([[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
# b = [x for x in range(5)]
# c = deque(a)
# # c.extend(b)
# d = np.repeat(c, 2, axis=0)
# e = np.repeat(b, 2, axis=0)
# # print(len(c)) 
# print(d)
# print(e)
# print(len(c))