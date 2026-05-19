#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include <type_traits>

#include "common/data.cuh"

static constexpr int RADIAL_KERNEL_PLUMMER = 0;
static constexpr int RADIAL_KERNEL_QUARTIC_PLUMMER = 1;

template<int radial_kernel_kind>
struct RadialKernel;

template<typename tvec>
__device__ __forceinline__ tvec radial_kernel_rsqrt(tvec x) {
    if constexpr (std::is_same_v<tvec, float>) {
        return rsqrtf(x);
    } else {
        return rsqrt(x);
    }
}

template<typename tvec>
__device__ __forceinline__ tvec radial_kernel_sqrt(tvec x) {
    if constexpr (std::is_same_v<tvec, float>) {
        return sqrtf(x);
    } else {
        return sqrt(x);
    }
}

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER> {
    template<typename tvec>
    struct Params {
        tvec softening;
        tvec softening2;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        const tvec softening = params[0];
        return Params<tvec>{softening, softening * softening};
    }

    template<typename tvec>
    __device__ __forceinline__ static tvec self_value(Params<tvec> params) {
        return tvec(1) / params.softening;
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs
    ) {
        // coeffs[n] = ((1/r) d/dr)^n K(r) = 2^n d^n K / d(r^2)^n
        // for the Plummer-softened radial kernel K(r) = 1 / sqrt(r^2 + eps^2).
        tvec rinv = radial_kernel_rsqrt(r2 + params.softening2);
        tvec rinv2 = rinv * rinv;
        coeffs[0] = rinv;

        #pragma unroll
        for(int n = 1; n <= p; n++) {
            coeffs[n] = -tvec(2*n - 1) * coeffs[n-1] * rinv2;
        }
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_QUARTIC_PLUMMER> {
    template<typename tvec>
    struct Params {
        tvec softening;
        tvec softening4;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        const tvec softening = params[0];
        const tvec softening2 = softening * softening;
        return Params<tvec>{softening, softening2 * softening2};
    }

    template<typename tvec>
    __device__ __forceinline__ static tvec self_value(Params<tvec> params) {
        return tvec(1) / params.softening;
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs
    ) {
        // coeffs[n] = ((1/r) d/dr)^n K(r) = 2^n d^n K / d(r^2)^n
        // for the quartic Plummer kernel K(r) = 1 / (r^4 + eps^4)^(1/4).
        const tvec q = r2*r2 + params.softening4;
        const tvec qinv = tvec(1) / q;

        tvec scale = radial_kernel_rsqrt(radial_kernel_sqrt(q));
        tvec poly[p + 2];

        #pragma unroll
        for(int k = 0; k < p + 2; k++) {
            poly[k] = tvec(0);
        }
        poly[0] = tvec(1);

        #pragma unroll
        for(int n = 0; n <= p; n++) {
            tvec pval = tvec(0);
            tvec r2pow = tvec(1);

            #pragma unroll
            for(int k = 0; k <= p; k++) {
                if(k <= n) {
                    pval += poly[k] * r2pow;
                }
                r2pow *= r2;
            }
            coeffs[n] = pval * scale;

            if(n < p) {
                tvec next[p + 2];
                #pragma unroll
                for(int k = 0; k < p + 2; k++) {
                    next[k] = tvec(0);
                }

                #pragma unroll
                for(int k = 0; k <= p; k++) {
                    if(k <= n) {
                        if(k > 0) {
                            next[k - 1] += tvec(2) * params.softening4 * tvec(k) * poly[k];
                        }
                        next[k + 1] += (tvec(2*k) - tvec(4*n + 1)) * poly[k];
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
