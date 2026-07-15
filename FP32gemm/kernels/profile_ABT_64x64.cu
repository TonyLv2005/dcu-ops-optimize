#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define CHECK(cmd) do { hipError_t err=cmd; if(err!=hipSuccess) { printf("HIP error: %s at %d\n", hipGetErrorString(err), __LINE__); exit(1); }}while(0)
// Include the dispatch file, renaming its main to avoid conflict
#define main __main_dispatch
#include "gemm_ABT_dispatch.cu"
#undef main

int main() {
    CHECK(hipSetDevice(7));
    const int MM = 4096, NN = 256, KK_ = 3072;
    uint16_t *hA = (uint16_t*)malloc(MM * KK_ * sizeof(uint16_t));
    float *hB = (float*)malloc(NN * KK_ * sizeof(float));
    srand(42);
    for (int i = 0; i < MM * KK_; ++i) {
        float f = (float)(rand()%1000)/100.0f - 5.0f;
        uint32_t u; memcpy(&u, &f, 4);
        uint16_t h = (u>>16)&0xFFFF;
        if (u&0x0000FFFF && ((u>>16)&0x7FFF)!=0x7FFF) h += 1;
        hA[i] = h;
    }
    for (int i = 0; i < NN * KK_; ++i)
        hB[i] = (float)(rand()%1000)/100.0f - 5.0f;
    uint16_t *dA; float *dB, *dC;
    CHECK(hipMalloc(&dA, MM * KK_ * sizeof(uint16_t)));
    CHECK(hipMalloc(&dB, NN * KK_ * sizeof(float)));
    CHECK(hipMalloc(&dC, MM * NN * sizeof(float)));
    CHECK(hipMemcpy(dA, hA, MM * KK_ * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, NN * KK_ * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemset(dC, 0, MM * NN * sizeof(float)));
    dim3 grid((NN+63)/64, (MM+63)/64, 3);  // (4,64,3) for full coverage
    gemm_ABT_kslice_64x64_lds_B_k64_d<1024><<<grid, 256>>>(dA, dB, dC, MM, NN, KK_);
    CHECK(hipDeviceSynchronize());
    printf("Done. grid=(%d,%d,%d) M=%d\n", grid.x, grid.y, grid.z, MM);
    free(hA); free(hB); CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dC));
    return 0;
}
