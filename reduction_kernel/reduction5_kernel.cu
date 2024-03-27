#include <cuda_runtime.h>
#include <stdio.h>

// 使用原子操作的归约，必须在内核外把 out 初始化为 0
__global__ void
Reduction5_kernel(int* out, const int* in, size_t N) {
   const int tid = threadIdx.x;
   int partialSum = 0;
   for (size_t i = blockIdx.x * blockDim.x + tid;
        i < N;
        i += blockDim.x * gridDim.x) {
      partialSum += in[i];
   }
   atomicAdd(out, partialSum);
}

void Reduction5(int* answer, int* partial, const int* in, size_t N, int numBlocks, int numThreads) {
   cudaMemset(answer, 0, sizeof(int));
   Reduction5_kernel<<<numBlocks, numThreads>>>(answer, in, N);
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
   Reduction5(answer, partial, in, N, numBlocks, numThreads);

   // transfer output from device to host
   int h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
   printf("reduction sum: %d\n", h_answer[0]);
   return 0;
}
