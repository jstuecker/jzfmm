#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include "data.cuh"

#include <cmath>
using std::abs; // be sure we have a floating point and int compatible abs function

/* ---------------------------------------------------------------------------------------------- */
/*                                 Data type specific definitions                                 */
/* ---------------------------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------------------------- */
/*                                         Kahan Summation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<typename tvec>
__forceinline__ __device__ void kahan_add(
    tvec &sum, tvec add, tvec &c
) {
    if constexpr (std::is_floating_point_v<tvec>) {
        tvec y = add - c;
        tvec t = sum + y;
        c = (t - sum) - y;
        sum = t;
    } else {
        // for integer types we simply do a normal add, since they are anyways exact
        sum += add;
    }
}

template<int dim, typename tvec>
__forceinline__ __device__ void kahan_add_vec(
    Vec<dim,tvec> &sum, const Vec<dim,tvec> &add, Vec<dim,tvec> &c
) {
    #pragma unroll
    for(int i=0; i<dim; i++) {
        kahan_add<tvec>(sum[i], add[i], c[i]);
    }
}

template<bool use_kahan, int dim, typename tvec>
__forceinline__ __device__
void add_vec(Vec<dim,tvec>& sum, const Vec<dim,tvec>& add, Vec<dim,tvec>& c) {
    if(use_kahan)
        kahan_add_vec<dim,tvec>(sum, add, c);
    else
        sum += add;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          Integer Math                                          */
/* ---------------------------------------------------------------------------------------------- */

// Calculates integer division in round-up mode
__host__ __device__ __forceinline__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

template<typename tvec>
__host__ __device__ __forceinline__
tvec powi_upto6(tvec x, int n) {
    switch (n) {
        case 0: return tvec(1);
        case 1: return x;
        case 2: return x * x;
        case 3: { tvec x2 = x * x; return x2 * x; }
        case 4: { tvec x2 = x * x; return x2 * x2; }
        case 5: { tvec x2 = x * x; tvec x4 = x2 * x2; return x4 * x; }
        case 6: { tvec x2 = x * x; tvec x3 = x2 * x; return x3 * x3; }
        case 7: { tvec x2 = x * x; tvec x3 = x2 * x; return x3 * x3 * x; }
        default: // fallback if someone passes >7
            if constexpr (std::is_same_v<tvec, float>) {
                return powf(x, float(n));
            } else {
                return pow(x, double(n));
            }
    }
}

__device__ __forceinline__ float binomial(unsigned n, unsigned k) {
    if (k > n) return 0.f;
    float res = 1.f;
    for (unsigned i = 1; i <= k; i++) {
        res *= float(n - (k - i)) / float(i);
    }
    return res;
}

__host__ __device__ __forceinline__ constexpr int binomial_int(int n, int k) {
    int res = 1;
    #pragma unroll
    for(int i = 1; i <= k; i++) {
        res = (res * (n - k + i)) / i;
    }
    return res;
}

template<int dim>
__host__ __device__ __forceinline__ constexpr int ncomb_dim(int p) {
    if constexpr (dim == 2) {
        return (p + 1) * (p + 2) / 2;
    } else if constexpr (dim == 3) {
        return (p + 1) * (p + 2) * (p + 3) / 6;
    } else {
        // Theoretically the dim=2 and dim=3 specializations should be unnecessary...
        // However, sometimes arrays landed in local memory with the more general version below:
        return binomial_int(p + dim, dim);
    }
}

#define NCOMB(p, dim) ncomb_dim<dim>((p))

template<int dim>
__device__ __forceinline__ Vec<dim,int32_t> lvl_vec(const int level) {
    // Converts a node's or leaf's binary level to its level per dimension

    // CUDA's integer division does not what we want for negative numbers. 
    // e.g. -4/3 = -1 whereas what we want is python behaviour: -4//3 = -2
    // We add an offset to ensure that CUDA divides positive integers only:
    int olvl = (level + 2000*dim) / dim - 2000;
    int omod = level - olvl * dim;

    Vec<dim,int32_t> lvec;
    #pragma unroll
    for(int i=0; i<dim; i++) {
        lvec[i] = olvl + (omod >= (dim-i));
    }
    
    return lvec;
}

// Geometric levels include sign-spanning (unbounded) cells. Their expansions
// are unused: the opening criterion always descends into their children.
template<int dim, typename T>
__device__ __forceinline__ bool unbounded_node(int level) {
    constexpr int max_exp = std::is_same_v<T, float> ? 128 : 1024;
    // The largest per-axis level is ceil(level / dim).
    return level > dim * (max_exp - 1);
}

// Expansion normalization need not equal the geometric extent. The floor
// covers tiny/coincident cells; the ceiling keeps both 2^exponent and its
// reciprocal normal, allowing unchecked scaling in the hot M2L kernel.
template<typename T>
__device__ __forceinline__ int bounded_expansion_exponent(int exponent) {
    constexpr int min_exp = std::is_same_v<T, float> ? -32 : -256;
    constexpr int max_exp = std::is_same_v<T, float> ? 126 : 1022;
    return min(max(exponent, min_exp), max_exp);
}

template<int dim, typename T>
__device__ __forceinline__ int expansion_exponent(int level) {
    return bounded_expansion_exponent<T>(lvl_vec<dim>(level)[dim - 1]);
}

// Only call with an exponent giving a normal power of two.
template<typename T>
__device__ __forceinline__ T normal_pow2(int exponent) {
    if constexpr (std::is_same_v<T, float>)
        return __uint_as_float((static_cast<unsigned>(exponent) + 127u) << 23);
    else
        return __longlong_as_double((static_cast<unsigned long long>(exponent) + 1023ull) << 52);
}

// Scale a positive normal weight by 2^exponent (exponent <= 0).
// Flush weights below the normal range to zero; no general ldexp fallback.
// Adjusting exponent bits avoids an extra multiply for constant factorial weights.
template<typename T>
__device__ __forceinline__ T scale_weight_by_power_of_two(T value, int exponent) {
    if constexpr (std::is_same_v<T, float>) {
        const unsigned bits = __float_as_uint(value);
        const int result_exponent = int(bits >> 23) + exponent;
        const unsigned scaled_bits = bits + (static_cast<unsigned>(exponent) << 23);
        return result_exponent > 0 ? __uint_as_float(scaled_bits) : T(0);
    } else {
        const auto bits = static_cast<unsigned long long>(__double_as_longlong(value));
        const int result_exponent = int(bits >> 52) + exponent;
        const auto scaled_bits = bits + (static_cast<unsigned long long>(exponent) << 52);
        return result_exponent > 0 ? __longlong_as_double(scaled_bits) : T(0);
    }
}

template<typename T>
__device__ __forceinline__ int normal_ilogb(T positive_value) {
    if constexpr (std::is_same_v<T, float>)
        return int((__float_as_uint(positive_value) >> 23) & 255u) - 127;
    else
        return int((__double_as_longlong(positive_value) >> 52) & 2047ll) - 1023;
}

// Keep the exceptional path out of line so its temporaries do not compete
// with the expansion coefficients for registers in the usual case.
template <typename tvec>
__device__ __noinline__ tvec mulpow2_extreme(tvec val, int pow) {
    return ldexp(val, pow);
}

template <typename tvec>
__device__ __forceinline__ tvec mulpow2(tvec val, int pow) {
    if constexpr (std::is_same_v<tvec, float> || std::is_same_v<tvec, double>) {
        constexpr int min_exp = std::is_same_v<tvec, float> ? -126 : -1022;
        constexpr int max_exp = std::is_same_v<tvec, float> ? 127 : 1023;
        if (pow >= min_exp && pow <= max_exp)
            return val * normal_pow2<tvec>(pow);
        // Scale the value, not a separately rounded 2^pow: zero stays zero and
        // a small coefficient can survive a large positive scale difference.
        return mulpow2_extreme(val, pow);
    }
    else if constexpr (std::is_same_v<tvec, int32_t>)
        return pow >= 0 ? val << pow : val >> -pow;
    else if constexpr (std::is_same_v<tvec, int64_t>)
        return pow >= 0 ? val << pow : val >> -pow;
}

template<int dim, typename tvec>
__device__ __forceinline__ Vec<dim,tvec> mulpow2(Vec<dim,tvec> val, int pow) {
    #pragma unroll
    for(int i = 0; i < dim; i++)
        val[i] = mulpow2(val[i], pow);
    return val;
}

// Opening-criterion extents are divided by their maximum scale, so these
// exponents are nonpositive. Use the cheap, underflow-safe weight scaling.
template<typename tout, int dim, typename tin>
__device__ __forceinline__ Vec<dim,tout> nonpositive_exp2(const Vec<dim,tin>& exponent) {
    Vec<dim,tout> result;
    #pragma unroll
    for(int i = 0; i < dim; i++)
        result[i] = scale_weight_by_power_of_two(tout(1), int(exponent[i]));
    return result;
}

template<int dim, typename tvec>
__device__ __forceinline__ tvec absmax(const Vec<dim,tvec>& val) {
    tvec result = abs(val[0]);
    #pragma unroll
    for(int i = 1; i < dim; i++)
        result = max(result, abs(val[i]));
    return result;
}

#endif // COMMON_MATH_H
