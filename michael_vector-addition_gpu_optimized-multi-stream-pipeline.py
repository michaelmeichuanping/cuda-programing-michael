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

    # 之前是生成一个值比较整齐的向量，而现在改成了一个值更为随机的向量
    # x = np.arange(n).astype(np.int32)
    # y = 2 * x
    x = np.random.uniform(10,20,n)
    y = np.random.uniform(10,20,n)

    # manually copy data from the main memory to the GPU memory
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)


    # 在显卡设备上初始化一块GPU memory, for storing GPU计算结果, 以避免结果被回送到CPU
    # gpu_result = cuda.device_array(n)
    # CPU’s calculation result will still be in the main memory
    # cpu_result = np.empty(n)

    # 在显卡设备上初始化一块GPU memory, for storing GPU计算结果, 以避免结果被回送到CPU
    gpu_result = cuda.device_array(n)
    # 在显卡设备上再初始化一块GPU memory, for storing GPU计算结果(with multi-stream pipeline)
    z_streams_device = cuda.device_array(n)


    # calculate CUDA execution configuration [gridDim, blockDim]
    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    
    # get time-stamp when start
    start = time()
   
    # use GPU to do do vector sum, with execution configuration [19532, 1024]
    # gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result, n)
    # use data which has been manually copied into the GPU memory, instead of data in CPU memory
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
    
    # Device To Host
    default_stream_result = gpu_result.copy_to_host()

    cuda.synchronize()

    # print time consumed by GPU
    print("gpu vector add time " + str(time() - start))
    print("- gpu_result=", gpu_result)
    print("- default_stream_result=", default_stream_result)

    # resset timer for test of multi-stream-enabled configuration
    start = time()

    # below are codes for multi-stream pipeline, manually control everything!!!
    # use 5-stream
    number_of_streams = 5
    # 每个流处理的数据量为原来的 1/5, //: 做除法并只保留整数部分
    segment_size = n // number_of_streams

    # Initailize 5 x cuda streams
    stream_list = list()
    for i in range (0, number_of_streams):
        stream = cuda.stream()
        stream_list.append(stream)

    threads_per_block = 1024
    # 每个stream的处理的数据变为原来的1/5
    blocks_per_grid = math.ceil(segment_size / threads_per_block)
    streams_result = np.empty(n)

    # 启动多个stream
    for i in range(0, number_of_streams):
        # 传入不同的参数，让函数在不同的流执行
        
        # Host To Device
        # manually copy data from the main memory to the GPU memory
        x_i_device = cuda.to_device(x[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])
        y_i_device = cuda.to_device(y[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])

        # Kernel
        gpu_add[blocks_per_grid, threads_per_block, stream_list[i]](
                x_i_device,
                y_i_device,
                z_streams_device[i * segment_size : (i + 1) * segment_size],
                segment_size)

        # Device To Host
        streams_result[i * segment_size : (i + 1) * segment_size] = z_streams_device[i * segment_size : (i + 1) * segment_size].copy_to_host(stream=stream_list[i])

    cuda.synchronize()
    print("gpu streams vector add time " + str(time() - start))

    if (np.array_equal(default_stream_result, streams_result)):
        print("result correct")


if __name__ == "__main__":
    main()





