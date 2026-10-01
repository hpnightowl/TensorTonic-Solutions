#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Store N element-wise results in output; this kernel returns no value.
    for(int i=0; i < N; i++) {
        output[i] = input[i] < 0 ? 0: input[i];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}