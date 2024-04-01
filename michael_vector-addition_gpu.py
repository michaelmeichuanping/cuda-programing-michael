from numba import cuda
import numpy as np
import math
from time import time

# create a function for vector sum, run it over GPU
# Every GPU core just do the sum of 1 x component of the vector)
@cuda.jit
def gpu_add(a, b, result, n):
    # a, b为输入向量，result为输出向量
    # 向量维度为n
    # 得到当前thread的编号
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

def main():
    #  create 2 x vectors in 20M-dimension. covert them into int32
    #  Pass them to the function as the parameters
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 创建n维全0向量, 作为vector sum的初始值
    gpu_result = np.zeros(n)
    cpu_result = np.zeros(n)

    # calculate CUDA execution configuration [gridDim, blockDim]
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    
    # get time-stamp when start
    start = time()
   
    # use GPU to do do vector sum, with execution configuration [19532, 1024]
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result, n)
    cuda.synchronize()

    # print time consumed by GPU
    print("gpu vector add time " + str(time() - start))
    print("- gpu_result=", gpu_result)
    print("- gpu_result[1]=", gpu_result[1])

    # resset timer
    start = time()

    # use numpy function add() to do vector sum, by CPU
    cpu_result = np.add(x, y)

    # print time consumed by CPU
    print("cpu vector add time " + str(time() - start))
    print("- cpu_result=", cpu_result)
    print("- cpu_result[1]=", cpu_result[1])

    if (np.array_equal(cpu_result, gpu_result)):
        print("result correct")

if __name__ == "__main__":
    main()






