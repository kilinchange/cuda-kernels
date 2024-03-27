#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// 任意线程块大小的归约
__global__ void
Reduction6_kernel(int* out, const int* in, size_t N) {
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

   // Start the shared memory loop on the next power of 2 less
   // than the block size. If block size is not a power of 2,
   // accumulate the intermediate sums in the remainder range.
   int floorPow2 = blockDim.x;

   if (floorPow2 & (floorPow2 - 1)) {
      // 循环的作用是计算不大于 blockDim.x 的 2 的幂次
      while (floorPow2 & (floorPow2 - 1)) {
         floorPow2 &= floorPow2 - 1;
      }
      // 将超过 2 的幂次的线程计算得到的值累加到前面的线程里，以便后续进行对数归约
      if (tid >= floorPow2) {
         sPartials[tid - floorPow2] += sPartials[tid];
      }
      __syncthreads();
   }

   for (int activeThreads = floorPow2 >> 1;
        activeThreads;
        activeThreads >>= 1) {
      if (tid < activeThreads) {
         sPartials[tid] += sPartials[tid + activeThreads];
      }
      __syncthreads();
   }

   if (tid == 0) {
      out[blockIdx.x] = sPartials[0];
   }
}

void Reduction6(int* answer, int* partial, const int* in, size_t N, int numBlocks, int numThreads) {
   unsigned int sharedSize = numThreads * sizeof(int);
   Reduction6_kernel<<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
   Reduction6_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
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
      Reduction6(answer, partial, in, N, numBlocks, numThreads);
   }

   // transfer output from device to host
   int h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
   printf("reduction sum: %d\n", h_answer[0]);
   return 0;
}
