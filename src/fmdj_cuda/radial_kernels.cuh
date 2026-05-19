#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include "common/data.cuh"

static constexpr int RADIAL_KERNEL_PLUMMER = 0;
static constexpr int RADIAL_KERNEL_QUARTIC_PLUMMER = 1;

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

template<>
struct RadialKernel<RADIAL_KERNEL_QUARTIC_PLUMMER> {
    struct Params {
        float softening;
        float softening4;
    };

    __device__ __forceinline__ static Params make_params(const float* params) {
        const float softening = params[0];
        const float softening2 = softening * softening;
        return Params{softening, softening2 * softening2};
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
        // for the quartic Plummer kernel K(r) = 1 / (r^4 + eps^4)^(1/4).
        const float q = r2*r2 + params.softening4;
        const float qinv = 1.0f / q;

        float scale = rsqrtf(sqrtf(q));
        float poly[p + 2];

        #pragma unroll
        for(int k = 0; k < p + 2; k++) {
            poly[k] = 0.0f;
        }
        poly[0] = 1.0f;

        #pragma unroll
        for(int n = 0; n <= p; n++) {
            float pval = 0.0f;
            float r2pow = 1.0f;

            #pragma unroll
            for(int k = 0; k <= p; k++) {
                if(k <= n) {
                    pval += poly[k] * r2pow;
                }
                r2pow *= r2;
            }
            coeffs[n] = pval * scale;

            if(n < p) {
                float next[p + 2];
                #pragma unroll
                for(int k = 0; k < p + 2; k++) {
                    next[k] = 0.0f;
                }

                #pragma unroll
                for(int k = 0; k <= p; k++) {
                    if(k <= n) {
                        if(k > 0) {
                            next[k - 1] += 2.0f * params.softening4 * k * poly[k];
                        }
                        next[k + 1] += (2.0f * k - (4.0f * n + 1.0f)) * poly[k];
                    }
                }

                #pragma unroll
                for(int k = 0; k < p + 2; k++) {
                    poly[k] = next[k];
                }
                scale *= qinv;
            }
        }
    }
};

#endif // RADIAL_KERNELS_H
