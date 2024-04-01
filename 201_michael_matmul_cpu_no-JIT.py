
import numpy as np
import time
from numba import jit
 
def mutiply(arr_a,arr_b):
    res = np.zeros((arr_a.shape[0], arr_b.shape[1]))
    for i in range(arr_a.shape[0]):
        for j in range(arr_b.shape[1]):
            for k in range(arr_b.shape[0]):
                res[i,j] += arr_a[i,k] * arr_b[k,j]
    return res
 
M = 60
N = 48
P = 40
A = np.random.random((M, N)) # 随机生成的 [M x N] 矩阵
B = np.random.random((N, P)) # 随机生成的 [N x P] 矩阵

start = time.time()
print("Result:\n",mutiply(A,B))
print("time required:  %s" %(time.time() - start))





