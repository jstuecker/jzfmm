#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

struct __align__(16) LocalExp {
    union {
        struct {
            float  pot;
            float3 grad;
        };

        float4 f4;
    };
};

__forceinline__ __device__ LocalExp GetForceAndPot(
    PosMass xmi, 
    PosMass xmj, 
    float softening2
) {
    float3 dx = xmj.pos - xmi.pos;
    float rinv = rsqrtf(norm2(dx) + softening2);
    float minvr = -xmj.mass * rinv;

    LocalExp loc = {
        minvr,
        (minvr * rinv * rinv) * dx
    };

    return loc;
}

__forceinline__ __device__ PosMass VJP_GFPhiToGXM(
    const PosMass xmi, const PosMass xmj, 
    const LocalExp gi, const LocalExp gj, 
    float epsilon2
) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    float3 dx = xmj.pos - xmi.pos;
    float r2 = norm2(dx);
    float rinv = r2 > 1e-10f * epsilon2 ? rsqrtf(r2 + epsilon2) : 0.f;
    float rinv2 = rinv*rinv;

    float f1 = -rinv*rinv2;
    float f2 = 3*rinv*rinv2*rinv2;

    float3 gm_diff = xmi.mass * gj.grad - xmj.mass * gi.grad;
    float fgdiff = f2*dot(gm_diff, dx) + f1 * (gi.pot * xmj.mass + gj.pot * xmi.mass);

    PosMass gxmi;

    gxmi.pos = f1 * gm_diff + fgdiff * dx;
    gxmi.mass = - f1 * dot(dx, gj.grad) - rinv * gj.pot;

    return gxmi;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void ForceAndPotential(
    const PosMass *xm,
    LocalExp *loc_out,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass xmi;
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ PosMass xmj_shared[];

    LocalExp loc_i = {0.f, 0.f, 0.f, 0.f};
    LocalExp loc_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            LocalExp loc_new = GetForceAndPot(xmi, xmj_shared[j], epsilon2);
            if(kahan)
                kahan_add_f4(loc_i.f4, loc_new.f4, loc_kahan.f4);
            else
                loc_i.f4 = loc_i.f4 + loc_new.f4;
        }
    }

    loc_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    if(ipart < n)
        loc_out[blockIdx.x * blockDim.x + threadIdx.x] = loc_i;
}

template <bool kahan>
__global__ void BwdForceAndPotential(
    const LocalExp *gloc,
    const PosMass *xm,
    PosMass *gxm,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    PosMass xmi;
    LocalExp gloc_i;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    if(ipart < n) {
        xmi = xm[ipart];
        gloc_i = gloc[ipart];
    }

    extern __shared__ PosMass xmj_shared[];
    LocalExp* gloc_j_shared = (LocalExp*) &xmj_shared[blockDim.x];

    PosMass gxm_i = {0.f, 0.f, 0.f, 0.f};
    PosMass gxm_i_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num) {
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
            gloc_j_shared[threadIdx.x] = gloc[jblock * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        for (int j = 0; j < num; j++) {
            PosMass gxm_inc = VJP_GFPhiToGXM(
                xmi, xmj_shared[j],
                gloc_i, gloc_j_shared[j],
                epsilon2
            );
            if(kahan)
                kahan_add_f4(gxm_i.f4, gxm_inc.f4, gxm_i_kahan.f4);
            else
                gxm_i.f4 = gxm_i.f4 + gxm_inc.f4;
        }
    }

    if(ipart < n)
        gxm[ipart] = gxm_i;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     Grouped force kernel                                       */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void GroupedForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PosMass* posm,
    // outputs:
    LocalExp* loc_out,
    // attributes:
    float softening,
    int max_leaf_size
) {
    float softening2 = softening * softening;

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    int num = prange.y-prange.x;

    // Precalculate layout for M2L interactions: blockdim.x -> (num, n_write) + residuals
    int n_write = blockDim.x / num;
    // Which particle I am writing to:
    int a_write = threadIdx.x % num;   
    // Where I would read from in the first iteration:
    int read_b_offset = threadIdx.x / num;
    // Flag residual threads as invalid
    // (To avoid divergence we let these follow allong the calculations, 
    //  but later discard their result)
    int valid = threadIdx.x < num * n_write;

    PosMass xaWrite = posm[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    LocalExp loc_a = {0.f,0.f,0.f,0.f};
    LocalExp loc_a_kahan = {0.f,0.f,0.f,0.f};

    extern __shared__ PosMass xm_b[];

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0) {
            xm_b[threadIdx.x] = posm[id];
        }
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            LocalExp loc_new = GetForceAndPot(xaWrite, xm_b[ib], softening2);
            if(kahan)
                kahan_add_f4(loc_a.f4, loc_new.f4, loc_a_kahan.f4);
            else
                loc_a.f4 = loc_a.f4 + loc_new.f4;
        }

        __syncthreads();
    }

    if(valid) {
        int iout = prange.x + a_write;
        atomicAdd(&loc_out[iout].pot, loc_a.pot);
        atomicAdd(&loc_out[iout].grad.x, loc_a.grad.x);
        atomicAdd(&loc_out[iout].grad.y, loc_a.grad.y);
        atomicAdd(&loc_out[iout].grad.z, loc_a.grad.z);
    }
}

#endif