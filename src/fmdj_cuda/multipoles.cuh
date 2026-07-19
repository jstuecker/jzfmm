#ifndef MULTIPOLES_H
#define MULTIPOLES_H

#include <math_constants.h>

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

// template<int pmax>
// __device__ __forceinline__ constexpr  int3 flat_to_multi(const int kflat) {
//     int i = 0, ksum, kz, ky;
//     #pragma unroll
//     for(ksum=0; ksum <= pmax; ksum++) {
//         int nadd = ((ksum+2)*(ksum+1)) >> 1;
//         if (i + nadd > kflat)
//             break;
//         i += nadd;
//     }
//     #pragma unroll
//     for(kz=0; kz <= ksum; kz++) {
//         int nadd = (ksum-kz+1);
//         if (i + nadd > kflat)
//             break;
//         i += nadd;
//     }
//     ky = kflat - i;

//     return int3{ksum-ky-kz, ky, kz};
// }

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

template<int dim, typename tvec>
__device__ __forceinline__ tvec multiindex_factorial(const int (&k)[dim]) {
    tvec f = tvec(1);
    #pragma unroll
    for(int d = 0; d < dim; d++) {
        f *= tvec(fact_upto6f(k[d]));
    }
    return f;
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
    evaluate_radial_kernel_derivatives<p,tvec>(radial_kernel_kind, r2, radial_kernel_params, G);
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
/*                                         M2M Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim=3, typename tvec>
__device__ __forceinline__ void shift_multipoles(Vec<NCOMB(p, dim),tvec>& mp, Vec<NCOMB(p, dim),tvec>& mp_out, const Vec<dim,tvec> dpos) {
    constexpr int ncomb = NCOMB(p, dim);
    // Shift in x
    #pragma unroll
    for(int ax=0; ax<dim; ax++) {
        // shift dimension by dimension
        for_each_multiindex<p,dim>([&](int iflat, int, int (&k)[dim]) {
            tvec mnew = tvec(0);
            #pragma unroll
            for(int i = 0; i <= p; i++) {
                if(i <= k[ax]) { // not in loop boundaries to avoid dynamic indexing / local memory
                    int kfrom[dim];
                    copy_multiindex<dim>(k, kfrom);
                    kfrom[ax] = i;

                    tvec coeff = tvec(binomial(k[ax], i));
                    mnew += coeff * powi_upto6(dpos[ax], k[ax] - i) * mp[multi_to_flat<dim>(kfrom)];
                }
            }

            mp_out[iflat] = mnew;
        });

        if(ax < dim - 1) {
            #pragma unroll
            for(int i=0; i<ncomb; i++)
                mp[i] = mp_out[i];
        }
    }
}

template<int p, int dim, typename tvec>
__global__ void SummarizeMultipoles(
    const int* __restrict__ isplit,
    const tvec* __restrict__ mp_in,
    const Vec<dim,tvec>* __restrict__ xnode,
    const Vec<dim,tvec>* __restrict__ xchild,
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
    
    Vec<ncomb,tvec> mp_sum;
    Vec<ncomb,tvec> mp_kahan;

    #pragma unroll
    for(int iM=0; iM < ncomb; iM++) {
        mp_sum[iM] = tvec(0);
        mp_kahan[iM] = tvec(0);
    }

    for(int ip = istart; ip < iend; ip++) {
        Vec<ncomb,tvec> mp_new_in;
        for(int iM=0; iM < ncomb; iM++) {
            if(iM < ncomb_in)
                mp_new_in[iM] = mp_in[ip * ncomb_in + iM];
            else
                mp_new_in[iM] = tvec(0);
        }

        Vec<ncomb,tvec> mp_new_out;

        shift_multipoles<p,dim,tvec>(mp_new_in, mp_new_out, xchild[ip] - xnode[inode]);

        if(kahan)
            kahan_add_vec<ncomb,tvec>(mp_sum, mp_new_out, mp_kahan);
        else {
            #pragma unroll
            for(int iM=0; iM < ncomb; iM++) {
                mp_sum[iM] += mp_new_out[iM];
            }
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
    const Vec<dim,tvec>* __restrict__ xnode,
    const Vec<dim,tvec>* __restrict__ xchild,
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
    
    Vec<dim,tvec> xn = xnode[inode];
    Vec<ncomb,tvec> loc_in;
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        Vec<dim,tvec> dpos = xchild[ichild] - xn;

        Vec<ncomb,tvec> loc_out;

        Vec<ncomb,tvec> loc_src;
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p,dim,tvec>(loc_src, loc_out, dpos);

        #pragma unroll
        for (int iM = 0; iM < min(ncomb, NCOMB(pout, dim)); iM++) {
            loc_child[ichild * NCOMB(pout, dim) + iM] = loc_out[iM];
        }
    }
}

template<int p, int dim, typename tvec>
__global__ void TranslateLocalToLocal_XVJP(
    const int* __restrict__ isplit,
    const tvec* __restrict__ loc_node,
    const Vec<dim,tvec>* __restrict__ xnode,
    const Vec<dim,tvec>* __restrict__ xchild,
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
    
    Vec<dim,tvec> xn = xnode[inode];
    Vec<ncomb,tvec> loc_in;
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        Vec<dim,tvec> dpos = xchild[ichild] - xn;

        Vec<ncomb,tvec> loc_child;

        Vec<ncomb,tvec> loc_src;
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p,dim,tvec>(loc_src, loc_child, dpos);

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
                    
                gxa += gloc_child[im] * loc_child[ib] * (ma + 1);
            });

            g_xchild[ichild][a] = gxa;
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                         M2L Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int p_extra_m2l, int dim, typename tvec>
struct GeneratedM2L {
    static constexpr bool available = false;
};

#include "generated/m2l_specializations.cuh"

template<int p, int p_extra_m2l, int dim, typename tvec>
__device__ __forceinline__ void m2l_translator(
    Vec<dim,tvec> dx,
    const Vec<NCOMB(p, dim),tvec>& Mp,
    Vec<NCOMB(p + p_extra_m2l, dim),tvec>& loc,
    int radial_kernel_kind,
    const tvec* radial_kernel_params
) {
    constexpr int p_local = p + p_extra_m2l;
    static_assert(p_local >= 0, "p + p_extra_m2l must be non-negative");
    constexpr int pD = p > p_local ? p : p_local;
    constexpr int ncombD = NCOMB(pD, dim);

    using SpecializedM2L = GeneratedM2L<p,p_extra_m2l,dim,tvec>;
    if constexpr (SpecializedM2L::available) {
        SpecializedM2L::apply(dx, Mp, loc, radial_kernel_kind, radial_kernel_params);
    } else {
        Vec<ncombD,tvec> Dn;
        setupDnG<pD,dim,tvec>(dx, radial_kernel_kind, radial_kernel_params, Dn);

        for_each_multiindex<p_local,dim>([&](int kflat, int ksum, int (&k)[dim]) {
        tvec Lnew = tvec(0);

        for_each_multiindex<p,dim>([&](int nflat, int nsum, int (&n)[dim]) {
            if(ksum + nsum <= pD) {
                if constexpr (dim == 3) {
                    // This specialization helps with keeping the arrays in registers
                    // for dim = 3 and p = 5. Why? I don't know. For that case it makes
                    // a 40x performance difference
                    const int dn[3] = {k[0] + n[0], k[1] + n[1], k[2] + n[2]};
                    const tvec Dnk = Dn[multi_to_flat<3>(dn)];
                    const tvec Mpn = Mp[nflat];
                    const tvec infvac = tvec(1) / tvec(fact3f(n[0], n[1], n[2]));
                    Lnew += Dnk * Mpn * infvac;
                } else {
                    // General case, stays in registers for most setups
                    int dn[dim];
                    #pragma unroll
                    for(int d = 0; d < dim; d++) {
                        dn[d] = k[d] + n[d];
                    }
                    tvec Dnk = Dn[multi_to_flat<dim>(dn)];
                    tvec Mpn = Mp[nflat];

                    const tvec infvac = tvec(1) / multiindex_factorial<dim,tvec>(n);
                    Lnew += Dnk * Mpn * infvac;
                }
            }
        });

        const tvec sign = ksum % 2 == 0 ? tvec(1) : tvec(-1);
        tvec fac;
        if constexpr (dim == 3)
            fac = sign / tvec(fact3f(k[0], k[1], k[2]));
        else
            fac = sign / multiindex_factorial<dim,tvec>(k);

        loc[kflat] += Lnew * fac;
        });
    }
}

#endif // MULTIPOLES_H
