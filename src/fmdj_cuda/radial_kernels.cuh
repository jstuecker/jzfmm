#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include "common/data.cuh"

static constexpr int RADIAL_KERNEL_PLUMMER = 0;

template<int radial_kernel_kind>
struct RadialKernel;

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER> {
    struct Params {
        float softening;
        float softening2;
    };

    __device__ __forceinline__ static Params make_params(const float* params) {
        const float softening = params[0];
        return Params{softening, softening * softening};
    }

    __device__ __forceinline__ static float self_value(Params params) {
        return 1.0f / params.softening;
    }

    template<int p>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        float r2,
        Params params,
        Vec<p+1,float>& coeffs
    ) {
        // coeffs[n] = ((1/r) d/dr)^n K(r) = 2^n d^n K / d(r^2)^n
        // for the Plummer-softened radial kernel K(r) = 1 / sqrt(r^2 + eps^2).
        float rinv = rsqrtf(r2 + params.softening2);
        float rinv2 = rinv * rinv;
        coeffs[0] = rinv;

        #pragma unroll
        for(int n = 1; n <= p; n++) {
            coeffs[n] = -(2*n - 1) * coeffs[n-1] * rinv2;
        }
    }
};

#endif // RADIAL_KERNELS_H
