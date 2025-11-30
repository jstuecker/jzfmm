#ifndef CUSTOM_JAX_MULTIPOLES_H
#define CUSTOM_JAX_MULTIPOLES_H

#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"

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
__device__ void setupDnG(float3 dx, float eps2, float* __restrict__ Dn) {
    float r2 = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;

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
                    const float xk = k == 0 ? dx.x : (k == 1 ? dx.y : dx.z);

                    const int ilast = multi_to_flat(nx - 1*(k==0), ny - 1*(k==1), nz - 1*(k==2));
                    const int ilast2 = multi_to_flat(nx - 2*(k==0), ny - 2*(k==1), nz - 2*(k==2));

                    Dn[iflat] = xk*Dn[ilast] + (nk-1)*Dn[ilast2];
                    iflat -= 1;
                }
            }
        }

        Dn[0] = G[q];
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                 Multipole from particles kernel                                */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ void accumulate_m_and_mpos(float* smem, const PosMass& posm, float3 x0) {
    float m = posm.mass;
    float mx = posm.mass * (posm.x - x0.x);
    float my = posm.mass * (posm.y - x0.y);
    float mz = posm.mass * (posm.z - x0.z);

    float msum = warp_reduce_sum(m);
    float mxsum = warp_reduce_sum(mx);
    float mysum = warp_reduce_sum(my);
    float mzsum = warp_reduce_sum(mz);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&smem[0], msum);
        atomicAdd(&smem[1], mxsum);
        atomicAdd(&smem[2], mysum);
        atomicAdd(&smem[3], mzsum);
    }
}

template<int p>
__global__ void MultipolesFromParticles(
    const int* __restrict__ isplit,
    const PosMass* __restrict__ part_posm,
    float* __restrict__ mp_out,
    float3* __restrict__ xcom_out,
    bool around_com
) {
    constexpr int ncomb = NCOMB(p);

    int inode = blockIdx.x;
    int ipart_start = isplit[inode];
    int ipart_end = isplit[inode + 1];

    if (ipart_start >= ipart_end) {
        for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
            mp_out[inode * ncomb + iM] = 0.f;
        }
        if (threadIdx.x == 0) {
            xcom_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        }
        return;
    }

    PosMass p0 = part_posm[ipart_start];
    float3 x0 = {p0.x, p0.y, p0.z};

    __shared__ float mp[ncomb];
    #pragma unroll
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp[iM] = 0.0f;
    }
    __syncthreads();

    float3 center;
    if(around_com) {
        // First pass: compute total mass and center of mass
        for (int ioff = ipart_start; ioff < ipart_end; ioff += blockDim.x) {
            PosMass posm;
            if (ioff + threadIdx.x < ipart_end)
                posm = part_posm[ioff + threadIdx.x];
            else
                posm = PosMass{0.f, 0.f, 0.f, 0.f};
            
            accumulate_m_and_mpos(&mp[0], posm, x0);
        }
        __syncthreads();
        center = { mp[1] / mp[0] + x0.x, mp[2] / mp[0] + x0.y, mp[3] / mp[0] + x0.z};
        __syncthreads();
        if (threadIdx.x == 0) {
            mp[0] = 0.f; mp[1] = 0.f; mp[2] = 0.f; mp[3] = 0.f;
        }
        __syncthreads();
    }
    else {
        float3 x1 = part_posm[ipart_end - 1].pos;
        
        center = get_common_node(x0, x1).center;
    }

    // Second pass: compute multipoles around center of mass
    for (int ioff = ipart_start; ioff < ipart_end; ioff += blockDim.x) {
        PosMass dpos;
        if (ioff + threadIdx.x < ipart_end) {
            PosMass posm = part_posm[ioff + threadIdx.x];
            dpos = PosMass{ posm.x - center.x, posm.y - center.y, posm.z - center.z, posm.mass};
        }
        else
            dpos = PosMass{0.f, 0.f, 0.f, 0.f};

        int kflat = 0;
        #pragma unroll
        for(int ksum = 0; ksum <= p; ksum++) {
            #pragma unroll
            for(int kz = 0; kz <= ksum; kz++) {
                #pragma unroll
                for(int ky = 0; ky <= ksum - kz; ky++) {
                    const int kx = ksum - ky - kz;

                    float mnew = dpos.mass * powf(dpos.x, kx) * powf(dpos.y, ky) * powf(dpos.z, kz);

                    float msum = warp_reduce_sum(mnew);
                    if ((threadIdx.x & 31) == 0)
                        atomicAdd(&mp[kflat], msum);

                    kflat += 1;
                }
            }
        }
    }

    __syncthreads();

    // Write output
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp_out[inode * ncomb + iM] = mp[iM];
    }
    if (threadIdx.x == 0) {
        xcom_out[inode] = center;
    }
}

__global__ void CenterOfMass(
    const int* __restrict__ isplit,
    const float3* __restrict__ pos,
    const float* __restrict__ mass,
    float3* __restrict__ com_out,
    int nnodes,
    bool kahan
) {
    int inode = blockIdx.x * blockDim.x + threadIdx.x;

    if (inode >= nnodes)
        return;

    int istart = isplit[inode], iend = isplit[inode + 1];

    if(istart >= iend) {
        com_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        return;
    }
    
    float4 mp_sum = {0.f, 0.f, 0.f, 0.f};
    float4 mp_kahan = {0.f, 0.f, 0.f, 0.f};

    for(int ip = istart; ip < iend; ip++) {
        float m = mass[ip];
        float4 mpnew = {m, pos[ip].x * m, pos[ip].y * m, pos[ip].z * m};

        kahan_add_f4(mp_sum, mpnew, mp_kahan);
    }

    if (mp_sum.x > 0.f) {
        com_out[inode] = (1.f / mp_sum.x) * make_float3(mp_sum.y, mp_sum.z, mp_sum.w);
    }
    else {
        com_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         M2M Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p>
__device__ __forceinline__ void shift_multipoles(float *mp, float *mp_out, const float3 dpos) {

    // Shift in x
    int iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float mnew = 0.f;
                #pragma unroll
                for(int i = 0; i <= kx; i++) {
                    float coeff = binomial(kx, i);
                    mnew += coeff * powi_upto6(dpos.x, kx - i) * mp[multi_to_flat(i, ky, kz)];
                }

                mp_out[iflat] = mnew;
                iflat += 1;
            }
        }
    }
        
    // Shift in y
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float mnew = 0.f;
                #pragma unroll
                for(int j = 0; j <= ky; j++) {
                    float coeff = binomial(ky, j);
                    mnew += coeff * powi_upto6(dpos.y, ky - j) * mp_out[multi_to_flat(kx, j, kz)];
                }

                mp[iflat] = mnew;
                iflat += 1;
            }
        }
    }

    // Shift in z
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;
                float mnew = 0.f;
                #pragma unroll
                for(int l = 0; l <= kz; l++) {
                    float coeff = binomial(kz, l);
                    mnew += coeff * powi_upto6(dpos.z, kz - l) * mp[multi_to_flat(kx, ky, l)];
                }

                mp_out[iflat] = mnew;
                iflat += 1;
            }
        }
    }
}


template<int p>
__global__ void CoarsenMultipoles(
    const int* __restrict__ isplit,
    const float* __restrict__ mp_values,
    const float3* __restrict__ xcent,
    float* __restrict__ mp_out,
    float3* __restrict__ xcent_out,
    bool around_com
) {
    constexpr int ncomb = NCOMB(p);

    int inode = blockIdx.x;
    int istart = isplit[inode];
    int iend = isplit[inode + 1];

    // If there are no children, write zeros and NaN center
    if (istart >= iend) {
        for (int i = threadIdx.x; i < ncomb; i += blockDim.x)
            mp_out[inode * ncomb + i] = 0.f;
        if (threadIdx.x == 0)
            xcent_out[inode] = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
        return;
    }

    float3 x0 = xcent[istart];

    __shared__ float mp[ncomb];
    #pragma unroll
    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp[iM] = 0.0f;
    }
    __syncthreads();

    float3 center;

    if(around_com) {
        // First pass: compute total mass and center of mass
        for (int ioff = istart; ioff < iend; ioff += blockDim.x) {
            int index = ioff + threadIdx.x;
            PosMass posm;
            if (index < iend)
                posm = PosMass{
                    xcent[index].x, xcent[index].y, xcent[index].z,
                    mp_values[index * ncomb + 0]
                };
            else
                posm = PosMass{0.f, 0.f, 0.f, 0.f};
            
            accumulate_m_and_mpos(&mp[0], posm, x0);
        }
        __syncthreads();
        center = { mp[1] / mp[0] + x0.x, mp[2] / mp[0] + x0.y, mp[3] / mp[0] + x0.z };
        __syncthreads();
        if (threadIdx.x == 0) {
            mp[0] = 0.f; mp[1] = 0.f; mp[2] = 0.f; mp[3] = 0.f;
        }
        __syncthreads();
    }
    else {
        float3 x1 = xcent[iend - 1];

        center = get_common_node(x0, x1).center;
    }

    // Second pass: for each child shift its multipoles to the inode center and accumulate
    for (int ioff = istart; ioff < iend; ioff += blockDim.x) {
        int index = ioff + threadIdx.x;
        // compute displacement from child center to inode com
        float3 dpos = make_float3(xcent[index].x-center.x, xcent[index].y-center.y, xcent[index].z-center.z);

        // load source multipoles
        // compute shifted multipoles into a small stack array
        float src[NCOMB(p)];
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            if(index < iend)
                src[i] = mp_values[index * ncomb + i];
            else
                src[i] = 0.f;
        }

        float moved[NCOMB(p)];
        
        shift_multipoles<p>(src, moved, dpos);

        for(int iM=0; iM < ncomb; iM++) {
            float val = index < iend ? moved[iM] : 0.f;
            float sum = warp_reduce_sum(val);
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&mp[iM], sum);
            }
        }
    }
    __syncthreads();

    // write inode center
    if (threadIdx.x == 0) {
        xcent_out[inode] = center;
    }

    for (int iM=threadIdx.x; iM < ncomb; iM += blockDim.x) {
        mp_out[inode * ncomb + iM] = mp[iM];
    }
}

template<int p>
__global__ void SummarizeMultipoles(
    const int* __restrict__ isplit,
    const float3* __restrict__ xnode,
    const float3* __restrict__ xchild,
    const float* __restrict__ mp_in,
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

template<int p>
__device__ __forceinline__ void shift_local_to_local(float *loc, float *loc_out, float3 dpos) {

    // Shift in x
    int iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float lnew = 0.f;
                #pragma unroll
                for(int i = kx; i <= p - ky - kz; i++) {
                    float coeff = binomial(i, kx);
                    lnew += coeff * powi_upto6(dpos.x, i - kx) * loc[multi_to_flat(i, ky, kz)];
                }

                loc_out[iflat] = lnew;
                iflat += 1;
            }
        }
    }
        
    // Shift in y
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;

                float lnew = 0.f;
                #pragma unroll
                for(int j = ky; j <= p - kx - kz; j++) {
                    float coeff = binomial(j, ky);
                    lnew += coeff * powi_upto6(dpos.y, j - ky) * loc_out[multi_to_flat(kx, j, kz)];
                }

                loc[iflat] = lnew;
                iflat += 1;
            }
        }
    }

    // Shift in z
    iflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;
                float lnew = 0.f;
                #pragma unroll
                for(int l = kz; l <= p - kx - ky; l++) {
                    float coeff = binomial(l, kz);
                    lnew += coeff * powi_upto6(dpos.z, l - kz) * loc[multi_to_flat(kx, ky, l)];
                }

                loc_out[iflat] = lnew;
                iflat += 1;
            }
        }
    }
}

template<int p>
__global__ void TranslateLocalToLocal(
    const int* __restrict__ isplit,
    const float* __restrict__ loc_node,
    const float3* __restrict__ xnode,
    const float3* __restrict__ xchild,
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
    
    float3 xn = xnode[inode];
    float loc_in[ncomb];
    #pragma unroll
    for (int iM = 0; iM < ncomb; iM++) {
        loc_in[iM] = loc_node[inode * ncomb + iM];
    }
    
    for(int ichild = istart; ichild < iend; ichild++) {
        float3 dpos = xchild[ichild] - xn;

        float loc_out[ncomb];

        float loc_src[ncomb];
        #pragma unroll
        for (int i = 0; i < ncomb; i++) {
            loc_src[i] = loc_in[i];
        }

        shift_local_to_local<p>(loc_src, loc_out, dpos);

        // write output local expansion
        // we allow to write only a subset of the components
        // for example:
        // p=0 will output only the potential
        // p=1 potential and force and
        // p=2 potential, force and hessian
        #pragma unroll
        for (int iM = 0; iM < min(ncomb, NCOMB(pout)); iM++) {
            loc_child[ichild * NCOMB(pout) + iM] = loc_out[iM];
        }
    }
}


/* ---------------------------------------------------------------------------------------------- */
/*                                         M2L Translation                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int p>
__device__ __forceinline__ void m2l_translator(
    float3 dx,
    const float* Mp,
    float* loc,
    float epsilon2
) {
    constexpr int ncomb = NCOMB(p);

    float Dn[ncomb];
    setupDnG<p>(dx, epsilon2, Dn);

    int kflat = 0;
    #pragma unroll
    for(int ksum = 0; ksum <= p; ksum++) {
        #pragma unroll
        for(int kz = 0; kz <= ksum; kz++) {
            #pragma unroll
            for(int ky = 0; ky <= ksum - kz; ky++) {
                const int kx = ksum - ky - kz;
                float Lnew = 0.f;

                int nflat = 0;
                #pragma unroll
                for (int nsum = 0; nsum <= p - ksum; nsum++) {
                    #pragma unroll
                    for (int nz = 0; nz <= nsum; nz++) {
                        #pragma unroll
                        for (int ny = 0; ny <= nsum - nz; ny++) {
                            const int nx = nsum - ny - nz;
                            
                            float Dnk = Dn[multi_to_flat(kx + nx, ky + ny, kz + nz)];
                            float Mpn = Mp[nflat];
                            
                            const float infvac = 1./fact3f(nx, ny, nz);
                            Lnew += Dnk * Mpn * infvac;
                            nflat += 1;
                        }
                    }
                }

                const float sign = ksum % 2 == 0 ? -1.f : 1.f;
                const float fac = sign / fact3f(kx, ky, kz);

                loc[kflat] += Lnew * fac;

                kflat += 1;
            }
        }
    }
}

#endif // CUSTOM_JAX_MULTIPOLES_H