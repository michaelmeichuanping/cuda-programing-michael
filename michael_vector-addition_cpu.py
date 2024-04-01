
from numba import cuda
import numpy as np
import math
from time import time


def main():
    #  create 2 x vectors in 20M-dimension. covert them into int32
    #  Pass them to the function as the parameters
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 创建n维全0向量, 作为vector sum的初始值
    cpu_result = np.zeros(n)

    # resset timer
    start = time()

    # use numpy function add() to do vector sum, by CPU
    cpu_result = np.add(x, y)

    # print time consumed by CPU
    print("cpu vector add time " + str(time() - start))
    print("- cpu_result=", cpu_result)
    print("- cpu_result[1]=", cpu_result[1])

if __name__ == "__main__":
    main()





