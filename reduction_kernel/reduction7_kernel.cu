#include <cuda_runtime.h>
#include <stdio.h>

// 带模板归约内核，适应任意数据类型的归约

// 为避免对齐相关的编译错误，声明了一个变长的共享内存
// （书中原话，没太理解，但测试使用 int 或者 float 都没问题，用 double 输出会是 0）
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
   //    extern __shared__ T sPartials[];
   ReductionType sum;
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
   float h_in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
   size_t N = 8;
   size_t partialN = N;
   int numBlocks = 2;
   int numThreads = 4;

   // malloc device memory
   float *answer, *partial, *in;
   cudaMalloc((void**)&answer, 1 * sizeof(float));
   cudaMalloc((void**)&partial, partialN * sizeof(float));
   cudaMalloc((void**)&in, N * sizeof(int));

   // transfer data from host to device
   cudaMemcpy(in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

   // invoke the kernel
   Reduction7<float, float>(answer, partial, in, N, numBlocks, numThreads);

   // transfer output from device to host
   float h_answer[1];
   cudaMemcpy(h_answer, answer, 1 * sizeof(float), cudaMemcpyDeviceToHost);
   printf("reduction sum: %f\n", h_answer[0]);
   return 0;
}
