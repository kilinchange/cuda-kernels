#include <cuda_runtime.h>
#include <stdio.h>

// 以循环展开的线程束同步作为结束的归约
__global__ void
Reduction2_kernel(int* out, const int* in, size_t N) {
   extern __shared__ int sPartials[];
   int sum = 0;
   const int tid = threadIdx.x;
   for (size_t i = blockIdx.x * blockDim.x + tid;
        i < N;
        i += blockDim.x * gridDim.x) {
      sum += in[i];
   }
   sPartials[tid] = sum;
   __syncthreads();

   // 利用线程束同步代码，减少了对数步长归约过程中的条件代码数量
   for (int activeThreads = blockDim.x >> 1;
        activeThreads > 32;
        activeThreads >>= 1) {
      if (tid < activeThreads) {
         sPartials[tid] += sPartials[tid + activeThreads];
      }
      __syncthreads();
   }
   // 线程块的活动线程数低于 warp_size(32)，无需调用 __syncthreads() 函数，因为 warp 是按照锁步方式执行每条指令(SIMD)
   if (threadIdx.x < 32) {
      volatile int* wsSum = sPartials;
      if (blockDim.x > 32) {
         wsSum[tid] += wsSum[tid + 32];
      }
      wsSum[tid] += wsSum[tid + 16];
      wsSum[tid] += wsSum[tid + 8];
      wsSum[tid] += wsSum[tid + 4];
      wsSum[tid] += wsSum[tid + 2];
      wsSum[tid] += wsSum[tid + 1];
      if (tid == 0) {
         volatile int* wsSum = sPartials;
         out[blockIdx.x] = wsSum[0];
      }
   }
}
