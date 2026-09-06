#ifndef MULTIPOLES_H
#define MULTIPOLES_H

#include <math_constants.h>
#include <type_traits>

#include "common/data.cuh"
#include "common/math.cuh"
#include "radial_kernels.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                       Multipole Indexing                                       */
/* ---------------------------------------------------------------------------------------------- */

template<int dim>
__device__ __forceinline__ constexpr int multi_to_flat(const int (&k)[dim]) {
    int p = 0;
    #pragma unroll
    for(int d = 0; d < dim; d++) {
        if(k[d] < 0)
            return 0;
        p += k[d];
    }

    if constexpr (dim == 2) {
        int off = (p * (p + 1)) / 2 + k[1];
        return off > 0 ? off : 0;
    } else if constexpr (dim == 3) {
        int npoff = ((p+2)*(p+1)*p) / 6; // offset of the p-th symmeric tensor
        int off = npoff + (k[2]*(2*p + 3 - k[2]))/2 + k[1];
        return off > 0 ? off : 0;
    } else {
        int off = binomial_int(p + dim - 1, dim);
        int remaining = p;

        #pragma unroll
        for(int d = dim - 1; d >= 1; d--) {
            for(int kd = 0; kd < k[d]; kd++) {
                off += binomial_int(remaining - kd + d - 1, d - 1);
            }
            remaining -= k[d];
        }

        return off > 0 ? off : 0;
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                   Iteration Helper functions                                   */
/* ---------------------------------------------------------------------------------------------- */

// calls a function with each multi-index vec(k) so that sum(k) == const.
template<int dim, int d, typename F>
__device__ __forceinline__ void for_each_multiindex_fixed_sum(int ksum, int remaining, int (&k)[dim], int& iflat, F&& f) {
    if constexpr (d == 0) {
        k[0] = remaining;
        f(iflat, ksum, k);
        iflat += 1;
    } else {
        #pragma unroll
        for(int kd = 0; kd <= remaining; kd++) {
            k[d] = kd;
            for_each_multiindex_fixed_sum<dim, d - 1>(ksum, remaining - kd, k, iflat, f);
        }
    }
}

// calls a function with each multi-index vec(k) so that sum(k) <= p
template<int p, int dim, typename F>
__device__ __forceinline__ void for_each_multiindex(F&& f) {
    int k[dim];
    int iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        for_each_multiindex_fixed_sum<dim, dim - 1>(ksum, ksum, k, iflat, f);
    }
}

template<int dim, int d, typename F>
__device__ __forceinline__ void for_each_multiindex_fixed_sum_reverse(int ksum, int remaining, int (&k)[dim], F&& f) {
    if constexpr (d == 0) {
        k[0] = remaining;
        f(ksum, k);
    } else {
        #pragma unroll
        for(int kd = remaining; kd >= 0; kd--) {
            k[d] = kd;
            for_each_multiindex_fixed_sum_reverse<dim, d - 1>(ksum, remaining - kd, k, f);
        }
    }
}

template<int dim>
__device__ __forceinline__ void copy_multiindex(const int (&src)[dim], int (&dst)[dim]) {
    #pragma unroll
    for(int d = 0; d < dim; d++) {
        dst[d] = src[d];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                  Derivatives of Radial Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim=3, typename tvec>
__device__ __forceinline__ void setupDnG(
    Vec<dim,tvec> dx,
    int radial_kernel_kind,
    const tvec* radial_kernel_params,
    Vec<NCOMB(p, dim),tvec>& Dn
) {
    // Recurrence formula by Tausch (2003)
    tvec r2 = dx.norm2();

    Vec<p+1,tvec> G;
    evaluate_radial_kernel_derivatives<p,tvec>(
        radial_kernel_kind, r2, radial_kernel_params, 0, G
    );
    Dn[0] = G[p];

    #pragma unroll
    for(int q=p-1; q >= 0; q--) {
        int iflat = NCOMB(p-q, dim)-1;
        int n[dim];
        #pragma unroll
        for(int nsum=p-q; nsum >= 1; nsum--) {
            for_each_multiindex_fixed_sum_reverse<dim, dim - 1>(nsum, nsum, n, [&](int, int (&n)[dim]) {
                int ax = 0;
                #pragma unroll
                for(int d = dim - 1; d >= 1; d--) {
                    if(n[d] > 0) {
                        ax = d;
                        break;
                    }
                }

                const int na = n[ax];
                const tvec xa = dx[ax];

                int ilast_k[dim];
                int ilast2_k[dim];
                copy_multiindex<dim>(n, ilast_k);
                copy_multiindex<dim>(n, ilast2_k);
                ilast_k[ax] -= 1;
                ilast2_k[ax] -= 2;

                Dn[iflat] = xa*Dn[multi_to_flat<dim>(ilast_k)] + (na-1)*Dn[multi_to_flat<dim>(ilast2_k)];
                iflat -= 1;
            });
        }

        Dn[0] = G[q];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                              Compile-time coefficient utilities                                */
/* ---------------------------------------------------------------------------------------------- */

template<int begin, int end, typename F>
__device__ __forceinline__ void ct_static_for(F&& f) {
    if constexpr (begin < end) {
        f(std::integral_constant<int, begin>{});
        ct_static_for<begin + 1, end>(static_cast<F&&>(f));
    }
}

template<int i, typename F>
__device__ __forceinline__ void ct_static_for_reverse(F&& f) {
    if constexpr (i >= 0) {
        f(std::integral_constant<int, i>{});
        ct_static_for_reverse<i - 1>(static_cast<F&&>(f));
    }
}

// Runtime-addressable C arrays can be placed wholesale in local memory. Compile-time-only access
// lets NVCC scalarize this structure into independent registers.
template<int n, typename tvec>
struct RegisterArray {
    tvec value;
    RegisterArray<n - 1, tvec> tail;

    template<int i>
    __device__ __forceinline__ tvec& get() {
        static_assert(i >= 0 && i < n);
        if constexpr (i == n - 1)
            return value;
        else
            return tail.template get<i>();
    }
};

template<typename tvec>
struct RegisterArray<0, tvec> {};

template<int iflat, int dim>
__host__ __device__ constexpr int ct_flat_degree() {
    static_assert(dim == 2 || dim == 3);
    int degree = 0;
    while(iflat >= NCOMB(degree, dim))
        degree++;
    return degree;
}

template<int iflat, int dim, int axis>
__host__ __device__ constexpr int ct_flat_component() {
    static_assert(dim == 2 || dim == 3);
    static_assert(axis >= 0 && axis < dim);
    constexpr int degree = ct_flat_degree<iflat, dim>();
    const int offset = degree == 0 ? 0 : NCOMB(degree - 1, dim);
    int remaining = iflat - offset;

    if constexpr (dim == 2) {
        if constexpr (axis == 1)
            return remaining;
        else
            return degree - remaining;
    } else {
        int kz = 0;
        while(remaining >= degree - kz + 1) {
            remaining -= degree - kz + 1;
            kz++;
        }
        if constexpr (axis == 2)
            return kz;
        else if constexpr (axis == 1)
            return remaining;
        else
            return degree - remaining - kz;
    }
}

template<int dim, int kx, int ky, int kz=0>
__host__ __device__ constexpr int ct_multi_to_flat() {
    constexpr int degree = kx + ky + kz;
    if constexpr (dim == 2) {
        return degree * (degree + 1) / 2 + ky;
    } else {
        const int offset = (degree + 2) * (degree + 1) * degree / 6;
        return offset + kz * (2 * degree + 3 - kz) / 2 + ky;
    }
}

__host__ __device__ constexpr int ct_factorial(int n) {
    int result = 1;
    for(int i = 2; i <= n; i++)
        result *= i;
    return result;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         M2M Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim=3, typename tvec>
__device__ __forceinline__ void shift_multipoles(
    RegisterArray<NCOMB(p, dim),tvec>& mp, const Vec<dim,tvec> dpos
) {
    constexpr int ncomb = NCOMB(p, dim);
    ct_static_for<0, dim>([&](auto ax_constant) {
        constexpr int ax = decltype(ax_constant)::value;
        // Descending degree keeps every source coefficient unmodified until it is consumed.
        ct_static_for_reverse<ncomb - 1>([&](auto iflat_constant) {
            constexpr int iflat = decltype(iflat_constant)::value;
            constexpr int kx = ct_flat_component<iflat, dim, 0>();
            constexpr int ky = ct_flat_component<iflat, dim, 1>();
            constexpr int kz = []() constexpr {
                if constexpr (dim == 3)
                    return ct_flat_component<iflat, dim, 2>();
                else
                    return 0;
            }();
            constexpr int kax = ax == 0 ? kx : (ax == 1 ? ky : kz);
            tvec mnew = tvec(0);

            ct_static_for<0, p + 1>([&](auto i_constant) {
                constexpr int i = decltype(i_constant)::value;
                if constexpr (i <= kax) {
                    constexpr int from_kx = ax == 0 ? i : kx;
                    constexpr int from_ky = ax == 1 ? i : ky;
                    constexpr int from_kz = ax == 2 ? i : kz;
                    constexpr int ifrom =
                        ct_multi_to_flat<dim, from_kx, from_ky, from_kz>();
                    constexpr int displacement_power = kax - i;
                    mnew += tvec(binomial(kax, i))
                        * powi_upto6(dpos[ax], displacement_power)
                        * mp.template get<ifrom>();
                }
            });
            mp.template get<iflat>() = mnew;
        });
    });
}

template<int p, int dim, typename tvec>
// One warp per block leaves NVCC free to favor registers over occupancy.
__global__ __launch_bounds__(32, 1) void SummarizeMultipoles(
    const int* __restrict__ isplit,
    const tvec* __restrict__ mp_in,
    const Node<dim,tvec>* __restrict__ nodes,
    const Node<dim,tvec>* __restrict__ children,
    tvec* __restrict__ mp_out,
    int nnodes,
    int p_in,
    bool kahan
) {
    constexpr int ncomb = NCOMB(p, dim);
    int ncomb_in = NCOMB(p_in, dim);

    int inode = blockIdx.x * blockDim.x + threadIdx.x;

    if (inode >= nnodes)
        return;

    int istart = isplit[inode], iend = isplit[inode + 1];

    if(istart >= iend) {
        for(int iM=0; iM < ncomb; iM++) {
            mp_out[inode * ncomb + iM] = tvec(0);
        }
        return;
    }

    // These long-lived accumulators intentionally reside in local memory. The translated child
    // state below stays in registers, where it is repeatedly reused by the recurrence.
    volatile tvec mp_sum[ncomb];
    volatile tvec mp_kahan[ncomb];

    #pragma unroll
    for(int iM=0; iM < ncomb; iM++) {
        mp_sum[iM] = tvec(0);
        mp_kahan[iM] = tvec(0);
    }

    const Node<dim,tvec> node = nodes[inode];
    const int parent_exp = lvl_vec<dim>(node.level)[dim - 1];
    for(int ichild = istart; ichild < iend; ichild++) {
        const Node<dim,tvec> child = children[ichild];
        const int child_exp = lvl_vec<dim>(child.level)[dim - 1];
        const int child_dexp = child_exp - parent_exp;
        RegisterArray<ncomb,tvec> mp_new;
        ct_static_for<0, ncomb>([&](auto iM_constant) {
            constexpr int iM = decltype(iM_constant)::value;
            constexpr int degree = ct_flat_degree<iM, dim>();
            if(iM < ncomb_in) {
                mp_new.template get<iM>() = mulpow2(
                    mp_in[ichild * ncomb_in + iM], degree * child_dexp
                );
            }
            else
                mp_new.template get<iM>() = tvec(0);
        });

        shift_multipoles<p,dim,tvec>(
            mp_new, mulpow2(child.center - node.center, -parent_exp)
        );

        if(kahan) {
            ct_static_for<0, ncomb>([&](auto iM_constant) {
                constexpr int iM = decltype(iM_constant)::value;
                tvec sum = mp_sum[iM];
                tvec compensation = mp_kahan[iM];
                kahan_add<tvec>(sum, mp_new.template get<iM>(), compensation);
                mp_sum[iM] = sum;
                mp_kahan[iM] = compensation;
            });
        } else {
            ct_static_for<0, ncomb>([&](auto iM_constant) {
                constexpr int iM = decltype(iM_constant)::value;
                mp_sum[iM] += mp_new.template get<iM>();
            });
        }
    }

    for(int iM=0; iM < ncomb; iM++) {
        mp_out[inode * ncomb + iM] = mp_sum[iM];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         L2L Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim=3, typename tvec>
__device__ __forceinline__ void shift_local_to_local_inplace(
    RegisterArray<NCOMB(p, dim),tvec>& loc, const Vec<dim,tvec> dpos
) {
    constexpr int ncomb = NCOMB(p, dim);
    ct_static_for<0, dim>([&](auto ax_constant) {
        constexpr int ax = decltype(ax_constant)::value;
        // Ascending degree keeps every higher-order source coefficient unmodified
        // until it has contributed to all lower-order outputs.
        ct_static_for<0, ncomb>([&](auto iflat_constant) {
            constexpr int iflat = decltype(iflat_constant)::value;
            constexpr int kx = ct_flat_component<iflat, dim, 0>();
            constexpr int ky = ct_flat_component<iflat, dim, 1>();
            constexpr int kz = []() constexpr {
                if constexpr (dim == 3)
                    return ct_flat_component<iflat, dim, 2>();
                else
                    return 0;
            }();
            constexpr int ksum = kx + ky + kz;
            constexpr int kax = ax == 0 ? kx : (ax == 1 ? ky : kz);
            constexpr int imax = p - ksum + kax;
            tvec lnew = tvec(0);

            ct_static_for<0, p + 1>([&](auto i_constant) {
                constexpr int i = decltype(i_constant)::value;
                if constexpr (i >= kax && i <= imax) {
                    constexpr int from_kx = ax == 0 ? i : kx;
                    constexpr int from_ky = ax == 1 ? i : ky;
                    constexpr int from_kz = ax == 2 ? i : kz;
                    constexpr int ifrom =
                        ct_multi_to_flat<dim, from_kx, from_ky, from_kz>();
                    constexpr int displacement_power = i - kax;
                    lnew += tvec(binomial(i, kax))
                        * powi_upto6(dpos[ax], displacement_power)
                        * loc.template get<ifrom>();
                }
            });
            loc.template get<iflat>() = lnew;
        });
    });
}


template<int p, int dim=3, typename tvec>
__device__ __forceinline__ void shift_local_to_local(Vec<NCOMB(p, dim),tvec>& loc, Vec<NCOMB(p, dim),tvec>& loc_out, Vec<dim,tvec> dpos) {
    constexpr int ncomb = NCOMB(p, dim);
    #pragma unroll
    for(int ax=0; ax<dim; ax++) {
        // Shift dimension by dimension
        for_each_multiindex<p,dim>([&](int iflat, int ksum, int (&k)[dim]) {
            tvec lnew = tvec(0);
            #pragma unroll
            for(int i = k[ax]; i <= p; i++) {
                if(i <= p - ksum + k[ax]) { // not in loop boundaries to avoid dynamic indexing / local memory
                    tvec coeff = tvec(binomial(i, k[ax]));
                    int kfrom[dim];
                    copy_multiindex<dim>(k, kfrom);
                    kfrom[ax] = i;
                    lnew += coeff * powi_upto6(dpos[ax], i - k[ax]) * loc[multi_to_flat<dim>(kfrom)];
                }
            }
            loc_out[iflat] = lnew;
        });

        if(ax < dim - 1) {
            #pragma unroll
            for(int i=0; i<ncomb; i++)
                loc[i] = loc_out[i]; // copy back output as input for next iteration.
        }
    }
}

template<int p, int dim, typename tvec>
__global__ void TranslateLocalToLocal(
    const int* __restrict__ isplit,
    const tvec* __restrict__ loc_node,
    const Node<dim,tvec>* __restrict__ nodes,
    const Node<dim,tvec>* __restrict__ children,
    tvec* __restrict__ loc_child,
    const int nnodes,
    const int pout
) {
    constexpr int ncomb = NCOMB(p, dim);
    
    int inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (inode >= nnodes)
        return;
    
    int istart = isplit[inode], iend = isplit[inode + 1];
    if (istart >= iend)
        return;
    
    const Node<dim,tvec> node = nodes[inode];
    const int parent_exp = lvl_vec<dim>(node.level)[dim - 1];
    RegisterArray<ncomb,tvec> loc_in;
    ct_static_for<0, ncomb>([&](auto iM_constant) {
        constexpr int iM = decltype(iM_constant)::value;
        loc_in.template get<iM>() = loc_node[inode * ncomb + iM];
    });
    
    for(int ichild = istart; ichild < iend; ichild++) {
        const Node<dim,tvec> child = children[ichild];
        const int child_exp = lvl_vec<dim>(child.level)[dim - 1];
        const int child_dexp = child_exp - parent_exp;
        Vec<dim,tvec> dpos = mulpow2(child.center - node.center, -parent_exp);

        RegisterArray<ncomb,tvec> loc_work;
        ct_static_for<0, ncomb>([&](auto iM_constant) {
            constexpr int iM = decltype(iM_constant)::value;
            loc_work.template get<iM>() = loc_in.template get<iM>();
        });

        shift_local_to_local_inplace<p,dim,tvec>(loc_work, dpos);

        ct_static_for<0, ncomb>([&](auto iM_constant) {
            constexpr int iM = decltype(iM_constant)::value;
            constexpr int degree = ct_flat_degree<iM, dim>();
            if(iM < NCOMB(pout, dim)) {
                loc_child[ichild * NCOMB(pout, dim) + iM] =
                    mulpow2(loc_work.template get<iM>(), degree * child_dexp);
            }
        });
    }
}

template<int p, int dim, typename tvec>
__global__ void TranslateLocalToLocal_XVJP(
    const int* __restrict__ isplit,
    const tvec* __restrict__ loc_node,
    const Node<dim,tvec>* __restrict__ nodes,
    const Node<dim,tvec>* __restrict__ children,
    const tvec* __restrict__ g_loc_child,
    Vec<dim,tvec>* __restrict__ g_xchild,
    const int nnodes,
    const int pout
) {
    constexpr int ncomb = NCOMB(p, dim);
    
    int inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (inode >= nnodes)
        return;
    
    int istart = isplit[inode], iend = isplit[inode + 1];
    if (istart >= iend)
        return;
    
    const Node<dim,tvec> node = nodes[inode];
    const int parent_exp = lvl_vec<dim>(node.level)[dim - 1];
    Vec<ncomb,tvec> loc_in;
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        const Node<dim,tvec> child = children[ichild];
        const int child_exp = lvl_vec<dim>(child.level)[dim - 1];
        const int child_dexp = child_exp - parent_exp;
        Vec<dim,tvec> dpos = mulpow2(child.center - node.center, -parent_exp);

        Vec<ncomb,tvec> loc_child;

        Vec<ncomb,tvec> loc_src;
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p,dim,tvec>(loc_src, loc_child, dpos);

        for_each_multiindex<p,dim>([&](int im, int msum, int (&)[dim]) {
            loc_child[im] = mulpow2(loc_child[im], msum * child_dexp);
        });

        Vec<ncomb,tvec> gloc_child;
        #pragma unroll
        for (int iM = 0; iM < NCOMB(pout, dim); iM++) {
            gloc_child[iM] = g_loc_child[ichild * NCOMB(pout, dim) + iM];
        }

        #pragma unroll
        for (int a=0; a < dim; a++) {
            tvec gxa = tvec(0);
            for_each_multiindex<p,dim>([&](int im, int msum, int (&m)[dim]) {
                const int ma = m[a];

                int b[dim];
                copy_multiindex<dim>(m, b);
                b[a] += 1;
                const int bsum = msum + 1;

                if((msum > pout) || (bsum > p))
                    return;

                int ib = multi_to_flat<dim>(b);
                    
                gxa += mulpow2(
                    gloc_child[im] * loc_child[ib] * tvec(ma + 1), -child_exp
                );
            });

            g_xchild[ichild][a] = gxa;
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                         M2L Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim, typename tvec>
__device__ __forceinline__ void m2l_translator(
    Vec<dim,tvec> dx,
    int source_exp,
    int target_exp,
    const Vec<NCOMB(p, dim),tvec>& Mp,
    Vec<NCOMB(p, dim),tvec>& loc,
    int radial_kernel_kind,
    const tvec* radial_kernel_params
) {
    constexpr int ncomb = NCOMB(p, dim);
    int scale_exp = max(source_exp, target_exp);
    const tvec dx_max = absmax(dx);
    if(dx_max != tvec(0))
        scale_exp = max(scale_exp, normal_ilogb(dx_max));

    dx = mulpow2(dx, -scale_exp);
    const int source_dexp = source_exp - scale_exp;
    const int target_dexp = target_exp - scale_exp;
    Vec<p + 1,tvec> G;
    evaluate_radial_kernel_derivatives<p,tvec>(
        radial_kernel_kind, dx.norm2(), radial_kernel_params, scale_exp, G
    );

    #pragma unroll
    for(int q = 0; q <= p; q++) {
        // handle overflows for small separations and large orders p ~ 7
        if(!isfinite(G[q]))
            G[q] = tvec(0);
    }

    RegisterArray<ncomb,tvec> Dn;
    Dn.template get<0>() = G[p];

    // Tausch recurrence in the same order used by the generated translators.
    ct_static_for_reverse<p - 1>([&](auto q_constant) {
        constexpr int q = decltype(q_constant)::value;
        constexpr int max_flat = NCOMB(p - q, dim) - 1;
        ct_static_for_reverse<max_flat>([&](auto iflat_constant) {
            constexpr int iflat = decltype(iflat_constant)::value;
            if constexpr (iflat > 0) {
                constexpr int kx = ct_flat_component<iflat, dim, 0>();
                constexpr int ky = ct_flat_component<iflat, dim, 1>();
                constexpr int kz = []() constexpr {
                    if constexpr (dim == 3)
                        return ct_flat_component<iflat, dim, 2>();
                    else
                        return 0;
                }();
                constexpr int axis = kz > 0 ? 2 : (ky > 0 ? 1 : 0);
                constexpr int count = axis == 2 ? kz : (axis == 1 ? ky : kx);
                constexpr int prev_kx = axis == 0 ? kx - 1 : kx;
                constexpr int prev_ky = axis == 1 ? ky - 1 : ky;
                constexpr int prev_kz = axis == 2 ? kz - 1 : kz;
                constexpr int previous =
                    ct_multi_to_flat<dim, prev_kx, prev_ky, prev_kz>();

                tvec value = dx[axis] * Dn.template get<previous>();
                if constexpr (count > 1) {
                    constexpr int prev2_kx = axis == 0 ? kx - 2 : kx;
                    constexpr int prev2_ky = axis == 1 ? ky - 2 : ky;
                    constexpr int prev2_kz = axis == 2 ? kz - 2 : kz;
                    constexpr int previous2 =
                        ct_multi_to_flat<dim, prev2_kx, prev2_ky, prev2_kz>();
                    value += tvec(count - 1) * Dn.template get<previous2>();
                }
                Dn.template get<iflat>() = value;
            }
        });
        Dn.template get<0>() = G[q];
    });

    constexpr int nscale = (p + 1) * (p + 2) / 2;
    // One combined source/target exponent per degree pair (21 at p=5).
    RegisterArray<nscale,int> scale_exponents;
    ct_static_for<0, p + 1>([&](auto kd_constant) {
        constexpr int kd = decltype(kd_constant)::value;
        ct_static_for<0, p - kd + 1>([&](auto nd_constant) {
            constexpr int nd = decltype(nd_constant)::value;
            constexpr int index = kd * (p + 1) - kd * (kd - 1) / 2 + nd;
            const int exponent = nd * source_dexp + kd * target_dexp;
            scale_exponents.template get<index>() = exponent;
        });
    });

    ct_static_for<0, ncomb>([&](auto kflat_constant) {
        constexpr int kflat = decltype(kflat_constant)::value;
        constexpr int kx = ct_flat_component<kflat, dim, 0>();
        constexpr int ky = ct_flat_component<kflat, dim, 1>();
        constexpr int kz = []() constexpr {
            if constexpr (dim == 3)
                return ct_flat_component<kflat, dim, 2>();
            else
                return 0;
        }();
        constexpr int kdegree = kx + ky + kz;
        tvec Lnew = tvec(0);

        ct_static_for<0, ncomb>([&](auto nflat_constant) {
            constexpr int nflat = decltype(nflat_constant)::value;
            constexpr int nx = ct_flat_component<nflat, dim, 0>();
            constexpr int ny = ct_flat_component<nflat, dim, 1>();
            constexpr int nz = []() constexpr {
                if constexpr (dim == 3)
                    return ct_flat_component<nflat, dim, 2>();
                else
                    return 0;
            }();
            constexpr int ndegree = nx + ny + nz;
            if constexpr (kdegree + ndegree <= p) {
                constexpr int dflat =
                    ct_multi_to_flat<dim, kx + nx, ky + ny, kz + nz>();
                constexpr int nfac =
                    ct_factorial(nx) * ct_factorial(ny) * ct_factorial(nz);
                constexpr double inv_nfac = 1.0 / static_cast<double>(nfac);
                constexpr int scale_index =
                    kdegree * (p + 1) - kdegree * (kdegree - 1) / 2 + ndegree;
                const tvec weight = scale_weight_by_power_of_two(
                    tvec(inv_nfac), scale_exponents.template get<scale_index>());
                // Reuse this product across target coefficients of the same degree.
                const tvec scaled_mp = Mp[nflat] * weight;
                Lnew += Dn.template get<dflat>() * scaled_mp;
            }
        });

        constexpr int kfac = ct_factorial(kx) * ct_factorial(ky) * ct_factorial(kz);
        constexpr int sign = kdegree % 2 == 0 ? 1 : -1;
        constexpr double scale = static_cast<double>(sign) / static_cast<double>(kfac);
        loc[kflat] += Lnew * tvec(scale);
    });
}

#endif // MULTIPOLES_H
