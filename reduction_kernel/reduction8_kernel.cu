#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cinttypes>

// 基于洗牌指令的归约(__shfl_xor)
__global__ void
Reduction8_kernel(int* out, const int* in, size_t N) {
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
      int mySum = wsSum[tid];
      mySum += __shfl_xor_sync(uint32_t(-1), mySum, 16);
      mySum += __shfl_xor_sync(uint32_t(-1), mySum, 8);
      mySum += __shfl_xor_sync(uint32_t(-1), mySum, 4);
      mySum += __shfl_xor_sync(uint32_t(-1), mySum, 2);
      mySum += __shfl_xor_sync(uint32_t(-1), mySum, 1);

      if (tid == 0) {
         out[blockIdx.x] = mySum;
      }
   }
}

void Reduction8(int* answer, int* partial, const int* in, size_t N, int numBlocks, int numThreads) {
   unsigned int sharedSize = numThreads * sizeof(int);
   Reduction8_kernel<<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
   Reduction8_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}

int main() {
   // malloc host memory
   size_t N = 1024;
   int *h_in = reinterpret_cast<int *>(std::malloc(N * sizeof(int)));
   std::generate(h_in, h_in + N, [n = 0] mutable
                 { return ++n; });

   size_t partialN = N;
   int numBlocks = 2;
   int numThreads = 4;

   // malloc device memory
   int *answer, *partial, *in;
   cudaMalloc((void**)&answer, 1 * sizeof(int));
   cudaMalloc((void**)&partial, partialN * sizeof(int));
   cudaMalloc((void**)&in, N * sizeof(int));

   // transfer data from host to device
   cudaMemcpy(in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

   // invoke the kernel
   for (int i = 0; i < 1000; ++i)
   {
      Reduction8(answer, partial, in, N, numBlocks, numThreads);
   }

   // transfer output from device to host
   int h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
   printf("reduction sum: %d\n", h_answer[0]);
   return 0;
}
