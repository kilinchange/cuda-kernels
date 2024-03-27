#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// 单遍归约内核
template <unsigned int numThreads>
__device__ void
Reduction4_LogStepShared(int *out, volatile int *partials)
{
    const int tid = threadIdx.x;
    if (numThreads >= 1024)
    {
        if (tid < 512)
        {
            partials[tid] += partials[tid + 512];
        }
        __syncthreads();
    }
    if (numThreads >= 512)
    {
        if (tid < 256)
        {
            partials[tid] += partials[tid + 256];
        }
        __syncthreads();
    }
    if (numThreads >= 256)
    {
        if (tid < 128)
        {
            partials[tid] += partials[tid + 128];
        }
        __syncthreads();
    }
    if (numThreads >= 128)
    {
        if (tid < 64)
        {
            partials[tid] += partials[tid + 64];
        }
        __syncthreads();
    }

    // warp synchronous at the end
    if (tid < 32)
    {
        if (numThreads >= 64)
        {
            partials[tid] += partials[tid + 32];
        }
        if (numThreads >= 32)
        {
            partials[tid] += partials[tid + 16];
        }
        if (numThreads >= 16)
        {
            partials[tid] += partials[tid + 8];
        }
        if (numThreads >= 8)
        {
            partials[tid] += partials[tid + 4];
        }
        if (numThreads >= 4)
        {
            partials[tid] += partials[tid + 2];
        }
        if (numThreads >= 2)
        {
            partials[tid] += partials[tid + 1];
        }
        if (tid == 0)
        {
            *out = partials[0];
        }
    }
}

// Global variable used by reduceSinglePass to count blocks
__device__ unsigned int retirementCount = 0;

template <unsigned int numThreads>
__global__ void
reduceSinglePass(int *out, int *partial,
                 const int *in, unsigned int N)
{
    extern __shared__ int sPartials[];
    unsigned int tid = threadIdx.x;
    int sum = 0;
    for (size_t i = blockIdx.x * numThreads + tid;
         i < N;
         i += numThreads * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1)
    {
        Reduction4_LogStepShared<numThreads>(&out[blockIdx.x], sPartials);
        return;
    }
    Reduction4_LogStepShared<numThreads>(&partial[blockIdx.x], sPartials);
    __shared__ bool lastBlock;

    // wait for outstanding memory instructions in this thread
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0)
    {
        unsigned int ticket = atomicAdd(&retirementCount, 1);

        // If the ticket ID is equal to the number of blocks,
        // we are the last block!
        lastBlock = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // One block performs the final log-step reduction
    if (lastBlock)
    {
        int sum = 0;
        for (size_t i = tid;
             i < gridDim.x;
             i += numThreads)
        {
            sum += partial[i];
        }
        sPartials[threadIdx.x] = sum;
        __syncthreads();
        Reduction4_LogStepShared<numThreads>(out, sPartials);
        retirementCount = 0;
    }
}

template <unsigned int numThreads>
void Reduction4_template(int *answer, int *partial, const int *in, size_t N, int numBlocks)
{
   reduceSinglePass<numThreads><<<
       numBlocks, numThreads, numThreads * sizeof(int)>>>(
       answer, partial, in, N);
}

void Reduction4(int *out, int *partial, const int *in, size_t N, int numBlocks, int numThreads)
{
    switch (numThreads)
    {
    case 1:
        return Reduction4_template<1>(out, partial, in, N, numBlocks);
    case 2:
        return Reduction4_template<2>(out, partial, in, N, numBlocks);
    case 4:
        return Reduction4_template<4>(out, partial, in, N, numBlocks);
    case 8:
        return Reduction4_template<8>(out, partial, in, N, numBlocks);
    case 16:
        return Reduction4_template<16>(out, partial, in, N, numBlocks);
    case 32:
        return Reduction4_template<32>(out, partial, in, N, numBlocks);
    case 64:
        return Reduction4_template<64>(out, partial, in, N, numBlocks);
    case 128:
        return Reduction4_template<128>(out, partial, in, N, numBlocks);
    case 256:
        return Reduction4_template<256>(out, partial, in, N, numBlocks);
    case 512:
        return Reduction4_template<512>(out, partial, in, N, numBlocks);
    case 1024:
        return Reduction4_template<1024>(out, partial, in, N, numBlocks);
    }
}

int main()
{
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
    cudaMalloc((void **)&answer, 1 * sizeof(int));
    cudaMalloc((void**)&partial, partialN * sizeof(int));
    cudaMalloc((void**)&in, N * sizeof(int));

    // transfer data from host to device
    cudaMemcpy(in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // invoke the kernel
    for (int i = 0; i < 1000; ++i)
    {
        Reduction4(answer, partial, in, N, numBlocks, numThreads);
    }

    // transfer output from device to host
    int h_answer[1];
    cudaMemcpy(h_answer, answer, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("reduction sum: %d\n", h_answer[0]);
    return 0;
}
