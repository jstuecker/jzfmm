#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include "common/math.cuh"

static constexpr int RADIAL_KERNEL_PLUMMER = 0;
static constexpr int RADIAL_KERNEL_PLUMMER_2D = 1;
static constexpr int RADIAL_KERNEL_SOFTENED_DISTANCE = 2;

// scale_exp is zero for direct interactions, or bounded_expansion_exponent
// for M2L: both 2^scale_exp and its reciprocal are normal floating-point values.
template<int radial_kernel_kind>
struct RadialKernel;

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        tvec rinv = rsqrt(scaled_r2 + softening * softening);
        tvec rinv2 = rinv * rinv;
        coeffs[0] = -rinv;

        #pragma unroll
        for(int n = 1; n <= p; n++) {
            coeffs[n] = -tvec(2*n - 1) * coeffs[n-1] * rinv2;
        }

        coeffs = coeffs * normal_pow2<tvec>(-scale_exp);
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER_2D> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        const tvec rsoft2 = scaled_r2 + softening * softening;
        const tvec sinv = tvec(1) / rsoft2;
        coeffs[0] = tvec(0.5) * log(rsoft2)
            + tvec(scale_exp) * tvec(0.6931471805599453094);

        if constexpr (p >= 1) {
            coeffs[1] = sinv;

            #pragma unroll
            for(int n = 2; n <= p; n++) {
                coeffs[n] = -tvec(2 * (n - 1)) * coeffs[n - 1] * sinv;
            }
        }
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        const tvec rsoft = sqrt(scaled_r2 + softening * softening);
        coeffs[0] = rsoft;

        if constexpr (p >= 1) {
            const tvec rinv = tvec(1) / rsoft;
            const tvec rinv2 = rinv * rinv;
            coeffs[1] = rinv;

            #pragma unroll
            for(int n = 2; n <= p; n++) {
                coeffs[n] = -tvec(2*n - 3) * coeffs[n - 1] * rinv2;
            }
        }

        coeffs = coeffs * normal_pow2<tvec>(scale_exp);
    }
};

template<int p, typename tvec>
__device__ __forceinline__ void evaluate_radial_kernel_derivatives(
    int radial_kernel_kind,
    tvec scaled_r2,
    const tvec* params,
    int scale_exp,
    Vec<p+1,tvec>& coeffs
) {
    // s = scaled_r2 = r^2 / R^2, R = 2^scale_exp
    // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
    switch(radial_kernel_kind) {
        case RADIAL_KERNEL_SOFTENED_DISTANCE: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
        case RADIAL_KERNEL_PLUMMER_2D: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_PLUMMER_2D>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_PLUMMER_2D>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
        case RADIAL_KERNEL_PLUMMER:
        default: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_PLUMMER>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_PLUMMER>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
    }
}

template<int radial_kernel_kind, typename tvec>
__device__ __forceinline__ tvec radial_kernel_value(
    tvec r2,
    typename RadialKernel<radial_kernel_kind>::template Params<tvec> params
) {
    Vec<1,tvec> coeffs;
    RadialKernel<radial_kernel_kind>::template r2_derivative_coeffs<0,tvec>(
        r2, params, coeffs
    );
    return coeffs[0];
}

#endif // RADIAL_KERNELS_H
