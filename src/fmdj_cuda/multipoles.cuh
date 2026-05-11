#ifndef MULTIPOLES_H
#define MULTIPOLES_H

#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"

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

template<int dim>
__device__ __forceinline__ void copy_multiindex(const int (&src)[dim], int (&dst)[dim]) {
    #pragma unroll
    for(int d = 0; d < dim; d++) {
        dst[d] = src[d];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                 Derivatives of Green's Function                                */
/* ---------------------------------------------------------------------------------------------- */

template<int p>
__device__ void setupGn(float r2, float eps2, float* __restrict__ G)
{
    // The derivatives of (1/r d/dr)^n G_0  with G_0 = 1/r
    float rinv = rsqrtf(r2 + eps2); 
    float rinv2 = rinv*rinv;
    G[0] = 1.0f * rinv;

    #pragma unroll
    for (int n = 1; n <= p; n++) {
        G[n] = -(2*n-1) * G[n-1] * rinv2;
    }
}

template<int p>
__device__ void setupDnG(Vec<3,float> dx, float eps2, float* __restrict__ Dn) {
    float r2 = dx.norm2();

    float G[p+1];
    setupGn<p>(r2, eps2, G);
    Dn[0] = G[p];

    #pragma unroll
    for(int q=p-1; q >= 0; q--) {
        int iflat = NCOMB(p-q)-1;
        #pragma unroll
        for(int nsum=p-q; nsum >= 1; nsum--) {
            #pragma unroll
            for(int nz=nsum; nz >= 0; nz--) {
                #pragma unroll
                for(int ny=nsum-nz; ny >= 0; ny--) {
                    const int nx = nsum - ny - nz;

                    const int k = nz > 0 ? 2 : (ny > 0 ? 1 : 0);
                    const int nk = nz > 0 ? nz : (ny > 0 ? ny : nx);
                    const float xk = k == 0 ? dx[0] : (k == 1 ? dx[1] : dx[2]);

                    const int ilast_k[3] = {nx - 1*(k==0), ny - 1*(k==1), nz - 1*(k==2)};
                    const int ilast2_k[3] = {nx - 2*(k==0), ny - 2*(k==1), nz - 2*(k==2)};

                    Dn[iflat] = xk*Dn[multi_to_flat<3>(ilast_k)] + (nk-1)*Dn[multi_to_flat<3>(ilast2_k)];
                    iflat -= 1;
                }
            }
        }

        Dn[0] = G[q];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         M2M Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p, int dim=3>
__device__ __forceinline__ void shift_multipoles(float *mp, float *mp_out, const Vec<dim,float> dpos) {
    // Shift in x
    #pragma unroll
    for(int ax=0; ax<dim; ax++) {
        // shift dimension by dimension
        for_each_multiindex<p,dim>([&](int iflat, int, int (&k)[dim]) {
            float mnew = 0.f;
            #pragma unroll
            for(int i = 0; i <= p; i++) {
                if(i <= k[ax]) { // not in loop boundaries to avoid dynamic indexing / local memory
                    int kfrom[dim];
                    copy_multiindex<dim>(k, kfrom);
                    kfrom[ax] = i;

                    float coeff = binomial(k[ax], i);
                    mnew += coeff * powi_upto6(dpos[ax], k[ax] - i) * mp[multi_to_flat<dim>(kfrom)];
                }
            }

            mp_out[iflat] = mnew;
        });

        #pragma unroll
        for(int i=0; i<NCOMB(p); i++)
            mp[i] = mp_out[i];
    }
}

template<int p>
__global__ void SummarizeMultipoles(
    const int* __restrict__ isplit,
    const float* __restrict__ mp_in,
    const Vec<3,float>* __restrict__ xnode,
    const Vec<3,float>* __restrict__ xchild,
    float* __restrict__ mp_out,
    int nnodes,
    int p_in,
    bool kahan
) {
    constexpr int ncomb = NCOMB(p);
    int ncomb_in = NCOMB(p_in);

    int inode = blockIdx.x * blockDim.x + threadIdx.x;

    if (inode >= nnodes)
        return;

    int istart = isplit[inode], iend = isplit[inode + 1];

    if(istart >= iend) {
        for(int iM=0; iM < ncomb; iM++) {
            mp_out[inode * ncomb + iM] = 0.f;
        }
        return;
    }
    
    float mp_sum[ncomb];
    float mp_kahan[ncomb];

    #pragma unroll
    for(int iM=0; iM < ncomb; iM++) {
        mp_sum[iM] = 0.f;
        mp_kahan[iM] = 0.f;
    }

    for(int ip = istart; ip < iend; ip++) {
        float mp_new_in[ncomb];
        for(int iM=0; iM < ncomb; iM++) {
            if(iM < ncomb_in)
                mp_new_in[iM] = mp_in[ip * ncomb_in + iM];
            else
                mp_new_in[iM] = 0.f;
        }

        float mp_new_out[ncomb];

        shift_multipoles<p>(mp_new_in, mp_new_out, xchild[ip] - xnode[inode]);

        if(kahan)
            kahan_add_array<ncomb>(mp_sum, mp_new_out, mp_kahan);
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


template<int p, int dim=3>
__device__ __forceinline__ void shift_local_to_local(float *loc, float *loc_out, Vec<3,float> dpos) {
    #pragma unroll
    for(int ax=0; ax<dim; ax++) {
        // Shift dimension by dimension
        for_each_multiindex<p,dim>([&](int iflat, int ksum, int (&k)[3]) {
            float lnew = 0.f;
            #pragma unroll
            for(int i = k[ax]; i <= p; i++) {
                if(i <= p - ksum + k[ax]) { // not in loop boundaries to avoid dynamic indexing / local memory
                    float coeff = binomial(i, k[ax]);
                    int kfrom[dim];
                    copy_multiindex<dim>(k, kfrom);
                    kfrom[ax] = i;
                    lnew += coeff * powi_upto6(dpos[ax], i - k[ax]) * loc[multi_to_flat<dim>(kfrom)];
                }
            }
            loc_out[iflat] = lnew;
        });

        #pragma unroll
        for(int i=0; i<NCOMB(p); i++)
            loc[i] = loc_out[i]; // copy back output as input for next iteration,.
    }
}

template<int p>
__global__ void TranslateLocalToLocal(
    const int* __restrict__ isplit,
    const float* __restrict__ loc_node,
    const Vec<3,float>* __restrict__ xnode,
    const Vec<3,float>* __restrict__ xchild,
    float* __restrict__ loc_child,
    const int nnodes,
    const int pout
) {
    constexpr int ncomb = NCOMB(p);
    
    int inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (inode >= nnodes)
        return;
    
    int istart = isplit[inode], iend = isplit[inode + 1];
    if (istart >= iend)
        return;
    
    Vec<3,float> xn = xnode[inode];
    float loc_in[ncomb];
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        Vec<3,float> dpos = xchild[ichild] - xn;

        float loc_out[ncomb];

        float loc_src[ncomb];
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p>(loc_src, loc_out, dpos);

        #pragma unroll
        for (int iM = 0; iM < min(ncomb, NCOMB(pout)); iM++) {
            loc_child[ichild * NCOMB(pout) + iM] = loc_out[iM];
        }
    }
}

template<int p>
__global__ void TranslateLocalToLocal_XVJP(
    const int* __restrict__ isplit,
    const float* __restrict__ loc_node,
    const Vec<3,float>* __restrict__ xnode,
    const Vec<3,float>* __restrict__ xchild,
    const float* __restrict__ g_loc_child,
    Vec<3,float>* __restrict__ g_xchild,
    const int nnodes,
    const int pout
) {
    constexpr int ncomb = NCOMB(p);
    
    int inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (inode >= nnodes)
        return;
    
    int istart = isplit[inode], iend = isplit[inode + 1];
    if (istart >= iend)
        return;
    
    Vec<3,float> xn = xnode[inode];
    float loc_in[ncomb];
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        Vec<3,float> dpos = xchild[ichild] - xn;

        float loc_child[ncomb];

        float loc_src[ncomb];
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p>(loc_src, loc_child, dpos);

        float gloc_child[ncomb];
        #pragma unroll
        for (int iM = 0; iM < NCOMB(pout); iM++) {
            gloc_child[iM] = g_loc_child[ichild * NCOMB(pout) + iM];
        }

        #pragma unroll
        for (int a=0; a < 3; a++) {
            float gxa = 0.f;
            for_each_multiindex<p,3>([&](int im, int msum, int (&m)[3]) {
                const int ma = m[a];

                int b[3] = {m[0], m[1], m[2]};
                b[a] += 1;
                const int bsum = msum + 1;

                if((msum > pout) || (bsum > p))
                    return;

                int ib = multi_to_flat<3>(b);
                    
                gxa += gloc_child[im] * loc_child[ib] * (ma + 1);
            });

            g_xchild[ichild][a] = gxa;
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                         M2L Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p>
__device__ __forceinline__ void m2l_translator(
    Vec<3,float> dx,
    const float* Mp,
    float* loc,
    float epsilon2
) {
    constexpr int ncomb = NCOMB(p);

    float Dn[ncomb];
    setupDnG<p>(dx, epsilon2, Dn);

    for_each_multiindex<p,3>([&](int kflat, int ksum, int (&k)[3]) {
        float Lnew = 0.f;

        int nflat = 0;
        for_each_multiindex<p,3>([&](int, int nsum, int (&n)[3]) {
            if(nsum <= p - ksum) {
                const int dn[3] = {k[0] + n[0], k[1] + n[1], k[2] + n[2]};
                float Dnk = Dn[multi_to_flat<3>(dn)];
                float Mpn = Mp[nflat];
                
                const float infvac = 1./fact3f(n[0], n[1], n[2]);
                Lnew += Dnk * Mpn * infvac;
                nflat += 1;
            }
        });

        const float sign = ksum % 2 == 0 ? -1.f : 1.f;
        const float fac = sign / fact3f(k[0], k[1], k[2]);

        loc[kflat] += Lnew * fac;
    });
}

#endif // MULTIPOLES_H
