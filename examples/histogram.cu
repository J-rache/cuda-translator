// Differential fixture for cuda-translator.
//
// histogram   : uses atomicAdd -> OUTSIDE the translatable subset. The tool
//               must reject it with an exact, actionable diagnostic, never
//               emit wrong-but-compiling OpenCL.
// bin_classify: data-dependent control flow + integer math entirely inside
//               the subset: per-element bin id and keep-flag, no atomics.
#include <cuda_runtime.h>

extern "C" __global__ void histogram(const int* input, int* bins, int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = input[i];
        if (v >= 0 && v < nbins) {
            atomicAdd(&bins[v], 1);
        }
    }
}

extern "C" __global__ void bin_classify(const int* input, int* bin_of, int* keep,
                                        int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = input[i];
        int in_range = (v >= 0) & (v < nbins);
        bin_of[i] = v * in_range;
        keep[i] = in_range;
    }
}
