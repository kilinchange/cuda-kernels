#include <cuda_runtime.h>
#include <stdio.h>

// 两遍规约内核
__global__ void Reduction1_kernel(int* out, const int* in, size_t N) {
   extern __shared__ int sPartials[];
   int sum = 0;
   const int tid = threadIdx.x;
   // 对输入数组落入每个线程中的元素进行求和
   for (size_t i = blockIdx.x * blockDim.x + tid;
        i < N;
        i += blockDim.x * gridDim.x) {
      sum += in[i];
   }
   sPartials[tid] = sum;
   __syncthreads();

   // 针对共享内存中的值执行对数步长的归约操作
   for (int activeThreads = blockDim.x >> 1;
        activeThreads;
        activeThreads >>= 1) {
      if (tid < activeThreads) {
         sPartials[tid] += sPartials[tid + activeThreads];
      }
      __syncthreads();
   }

   // 线程块输出值写入全局内存
   if (tid == 0) {
      out[blockIdx.x] = sPartials[0];
   }
}

void Reduction1(int* answer, int* partial, const int* in, size_t N, int numBlocks, int numThreads) {
   unsigned int sharedSize = numThreads * sizeof(int);
   Reduction1_kernel<<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
   Reduction1_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}

int main() {
   // malloc host memory
   int h_in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
   size_t N = 8;
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
   Reduction1(answer, partial, in, N, numBlocks, numThreads);

   // transfer output from device to host
   int h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
   printf("reduction sum: %d\n", h_answer[0]);
   return 0;
}
