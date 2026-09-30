// Host-app fixture: kernels + main() in one TU so nvcc links a real host exe.
// Used by cuda-translator to test fatbin extraction from PE host binaries.
// Runnable on any machine with an NVIDIA GPU + driver (self-verifying).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void saxpy(float alpha, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}

static unsigned int lcg_state = 20260929u;
static float next_rand(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (float)((lcg_state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
}

int main(void) {
    const int n = 1 << 20;
    const size_t bytes = (size_t)n * sizeof(float);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes);
    float *h_x = (float*)malloc(bytes), *h_y = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes), *h_o = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_x || !h_y || !h_c || !h_o) return 2;

    for (int i = 0; i < n; i++) {
        h_a[i] = next_rand(); h_b[i] = next_rand();
        h_x[i] = next_rand(); h_y[i] = next_rand();
    }

    float *d_a, *d_b, *d_c, *d_x, *d_y, *d_o;
    if (cudaMalloc(&d_a, bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_c, bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_x, bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_y, bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_o, bytes) != cudaSuccess) return 3;

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    saxpy<<<blocks, threads>>>(2.5f, d_x, d_y, d_o, n);

    cudaError_t e1 = cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaError_t e2 = cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s / %s\n", cudaGetErrorString(e1), cudaGetErrorString(e2));
        return 4;
    }

    int fails = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) fails++;
    }
    printf("VERIFY vector_add %s (%d mismatches of %d)\n", fails == 0 ? "OK" : "FAIL", fails, n);

    fails = 0;
    for (int i = 0; i < n; i++) {
        float ref = 2.5f * h_x[i] + h_y[i];
        if (h_o[i] != ref) fails++;
    }
    printf("VERIFY saxpy %s (%d mismatches of %d)\n", fails == 0 ? "OK" : "FAIL", fails, n);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_o);
    free(h_a); free(h_b); free(h_x); free(h_y); free(h_c); free(h_o);
    return 0;
}
