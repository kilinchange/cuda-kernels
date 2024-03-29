#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// reduction2_kernel 的扩展，模板化的完全展开的对数步长的归约
// 为什么要将 numThreads 作为模板参数，而不是直接通过 blockDim.x 获取呢？
template <unsigned int numThreads>
__global__ void
Reduction3_kernel(int* out, const int* in, size_t N) {
   extern __shared__ int sPartials[];
   const unsigned int tid = threadIdx.x;
   int sum = 0;
   for (size_t i = blockIdx.x * numThreads + tid;
        i < N;
        i += numThreads * gridDim.x) {
      sum += in[i];
   }
   sPartials[tid] = sum;
   __syncthreads();

   if (numThreads >= 1024)
   {
      if (tid < 512) {
         sPartials[tid] += sPartials[tid + 512];
      }
      __syncthreads();
   }
   if (numThreads >= 512) {
      if (tid < 256) {
         sPartials[tid] += sPartials[tid + 256];
      }
      __syncthreads();
   }
   if (numThreads >= 256) {
      if (tid < 128) {
         sPartials[tid] += sPartials[tid + 128];
      }
      __syncthreads();
   }
   if (numThreads >= 128) {
      if (tid < 64) {
         sPartials[tid] += sPartials[tid + 64];
      }
      __syncthreads();
   }

   // warp synchronous at the end
   if (tid < 32) {
      volatile int* wsSum = sPartials;
      if (numThreads >= 64) {
         wsSum[tid] += wsSum[tid + 32];
      }
      if (numThreads >= 32) {
         wsSum[tid] += wsSum[tid + 16];
      }
      if (numThreads >= 16) {
         wsSum[tid] += wsSum[tid + 8];
      }
      if (numThreads >= 8) {
         wsSum[tid] += wsSum[tid + 4];
      }
      if (numThreads >= 4) {
         wsSum[tid] += wsSum[tid + 2];
      }
      if (numThreads >= 2) {
         wsSum[tid] += wsSum[tid + 1];
      }
      if (tid == 0) {
         out[blockIdx.x] = wsSum[0];
      }
   }
}

template <unsigned int numThreads>
void Reduction3_template(int* answer, int* partial, const int* in, size_t N, int numBlocks) {
   Reduction3_kernel<numThreads><<<
       numBlocks, numThreads, numThreads * sizeof(int)>>>(
       partial, in, N);
   Reduction3_kernel<numThreads><<<
       1, numThreads, numThreads * sizeof(int)>>>(
       answer, partial, numBlocks);
}

void Reduction3(int* out, int* partial, const int* in, size_t N, int numBlocks, int numThreads) {
   switch (numThreads) {
      case 1:
         return Reduction3_template<1>(out, partial, in, N, numBlocks);
      case 2:
         return Reduction3_template<2>(out, partial, in, N, numBlocks);
      case 4:
         return Reduction3_template<4>(out, partial, in, N, numBlocks);
      case 8:
         return Reduction3_template<8>(out, partial, in, N, numBlocks);
      case 16:
         return Reduction3_template<16>(out, partial, in, N, numBlocks);
      case 32:
         return Reduction3_template<32>(out, partial, in, N, numBlocks);
      case 64:
         return Reduction3_template<64>(out, partial, in, N, numBlocks);
      case 128:
         return Reduction3_template<128>(out, partial, in, N, numBlocks);
      case 256:
         return Reduction3_template<256>(out, partial, in, N, numBlocks);
      case 512:
         return Reduction3_template<512>(out, partial, in, N, numBlocks);
      case 1024:
         return Reduction3_template<1024>(out, partial, in, N, numBlocks);
   }
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
      Reduction3(answer, partial, in, N, numBlocks, numThreads);
   }

   // transfer output from device to host
   int h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
   printf("reduction sum: %d\n", h_answer[0]);
   return 0;
}
