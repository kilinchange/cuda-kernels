#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// 带模板归约内核，适应任意数据类型的归约

// 为避免对齐相关的编译错误，声明了一个变长的共享内存
// （书中原话，没太理解，不这样也不会错==）
template <class T>
struct SharedMemory {
   __device__ inline operator T*() {
      extern __shared__ int __smem[];
      return (T*)(void*)__smem;
   }

   __device__ inline operator const T*() const {
      extern __shared__ int __smem[];
      return (T*)(void*)__smem;
   }
};

template <typename ReductionType, typename T>
__global__ void
Reduction_templated(ReductionType* out, const T* in, size_t N) {
   SharedMemory<ReductionType> sPartials;
   // 并不会有问题==
   // extern __shared__ T sPartials[];
   ReductionType sum = 0;
   const int tid = threadIdx.x;
   for (size_t i = blockIdx.x * blockDim.x + tid;
        i < N;
        i += blockDim.x * gridDim.x) {
      sum += in[i];
   }
   sPartials[tid] = sum;
   __syncthreads();

   for (int activeThreads = blockDim.x >> 1;
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

template <typename ReductionType, typename T>
void Reduction7(ReductionType* answer, ReductionType* partial, const T* in, size_t N, int numBlocks, int numThreads) {
   unsigned int sharedSize = numThreads * sizeof(ReductionType);
   Reduction_templated<ReductionType, T><<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
   Reduction_templated<ReductionType, T><<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}

int main() {
   // malloc host memory
   size_t N = 1024;
   double *h_in = reinterpret_cast<double *>(std::malloc(N * sizeof(double)));
   std::generate(h_in, h_in + N, [n = 0] mutable
                 { return ++n; });

   size_t partialN = N;
   int numBlocks = 2;
   int numThreads = 4;

   // malloc device memory
   double *answer, *partial, *in;
   cudaMalloc((void **)&answer, 1 * sizeof(double));
   cudaMalloc((void **)&partial, partialN * sizeof(double));
   cudaMalloc((void **)&in, N * sizeof(double));

   // transfer data from host to device
   cudaMemcpy(in, h_in, N * sizeof(double), cudaMemcpyHostToDevice);

   // invoke the kernel
   for (int i = 0; i < 1000; ++i)
   {
      Reduction7<double, double>(answer, partial, in, N, numBlocks, numThreads);
   }

   // transfer output from device to host
   double h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(double), cudaMemcpyDeviceToHost);
   printf("reduction sum: %lf\n", h_answer[0]);
   return 0;
}
