#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

__forceinline__ __device__ LocalExp<3,float> GetForceAndPot(
    PosMass<3,float> xmi, 
    PosMass<3,float> xmj, 
    float softening2
) {
    Vec<3,float> dx = xmj.pos - xmi.pos;
    float rinv = rsqrtf(dx.norm2() + softening2);
    float minvr = -xmj.mass * rinv;

    LocalExp<3,float> loc = {
        minvr,
        (minvr * rinv * rinv) * dx
    };

    return loc;
}

__forceinline__ __device__ PosMass<3,float> VJP_GFPhiToGXM(
    const PosMass<3,float> xmi, const PosMass<3,float> xmj, 
    const LocalExp<3,float> gi, const LocalExp<3,float> gj, 
    float epsilon2
) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    Vec<3,float> dx = xmj.pos - xmi.pos;
    float r2 = dx.norm2();
    float rinv = r2 > 1e-10f * epsilon2 ? rsqrtf(r2 + epsilon2) : 0.f;
    float rinv2 = rinv*rinv;

    float f1 = -rinv*rinv2;
    float f2 = 3*rinv*rinv2*rinv2;

    Vec<3,float> gm_diff = xmi.mass * gj.grad - xmj.mass * gi.grad;
    float fgdiff = f2*gm_diff.dot(dx) + f1 * (gi.pot * xmj.mass + gj.pot * xmi.mass);

    PosMass<3,float> gxmi;

    gxmi.pos = f1 * gm_diff + fgdiff * dx;
    gxmi.mass = - f1 * dx.dot(gj.grad) - rinv * gj.pot;

    return gxmi;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan>
__global__ void ForceAndPotential(
    const PosMass<3,float> *xm,
    LocalExp<3,float> *loc_out,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass<3,float> xmi;
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ PosMass<3,float> xmj_shared[];

    LocalExp<3,float> loc_i = {0.f, 0.f, 0.f, 0.f};
    LocalExp<3,float> loc_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            LocalExp<3,float> loc_new = GetForceAndPot(xmi, xmj_shared[j], epsilon2);
            kahan_add_vec(loc_i.asvec, loc_new.asvec, loc_kahan.asvec);
        }
    }

    loc_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    if(ipart < n)
        loc_out[blockIdx.x * blockDim.x + threadIdx.x] = loc_i;
}

template <bool kahan>
__global__ void BwdForceAndPotential(
    const LocalExp<3,float> *gloc,
    const PosMass<3,float> *xm,
    PosMass<3,float> *gxm,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    PosMass<3,float> xmi;
    LocalExp<3,float> gloc_i;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    if(ipart < n) {
        xmi = xm[ipart];
        gloc_i = gloc[ipart];
    }

    extern __shared__ PosMass<3,float> xmj_shared[];
    LocalExp<3,float>* gloc_j_shared = (LocalExp<3,float>*) &xmj_shared[blockDim.x];

    PosMass<3,float> gxm_i = {0.f, 0.f, 0.f, 0.f};
    PosMass<3,float> gxm_i_kahan = {0.f, 0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num) {
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
            gloc_j_shared[threadIdx.x] = gloc[jblock * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        for (int j = 0; j < num; j++) {
            PosMass<3,float> gxm_inc = VJP_GFPhiToGXM(
                xmi, xmj_shared[j],
                gloc_i, gloc_j_shared[j],
                epsilon2
            );
            kahan_add_vec(gxm_i.asvec, gxm_inc.asvec, gxm_i_kahan.asvec);
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
    const PosMass<3,float>* posm,
    // outputs:
    LocalExp<3,float>* loc_out,
    // attributes:
    float softening
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

    PosMass<3,float> xaWrite = posm[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    LocalExp<3,float> loc_a = {0.f,0.f,0.f,0.f};
    LocalExp<3,float> loc_a_kahan = {0.f,0.f,0.f,0.f};

    extern __shared__ PosMass<3,float> xm_b[];

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0)
            xm_b[threadIdx.x] = posm[id];
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            LocalExp<3,float> loc_new = GetForceAndPot(xaWrite, xm_b[ib], softening2);
            add_vec<kahan>(loc_a.asvec, loc_new.asvec, loc_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    LocalExp<3,float>* loc_shared = (LocalExp<3,float>*) &xm_b[0];
    loc_shared[threadIdx.x] = loc_a;
    __syncthreads();

    if(read_b_offset == 0) {
        LocalExp<3,float> loc_cum = {0.f,0.f,0.f,0.f};
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(loc_cum.asvec, loc_shared[i*num + a_write].asvec, loc_a_kahan.asvec);

        loc_out[prange.x + a_write] = loc_cum;
    }
}

template <bool kahan>
__global__ void BwdGroupedForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PosMass<3,float>* posm,
    const LocalExp<3,float> *gloc,
    // outputs:
    PosMass<3,float>* gposm_out,
    // attributes:
    float softening
) {
    float softening2 = softening * softening;

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_nodes[nodeid], spl_nodes[nodeid + 1]};

    int num = prange.y-prange.x;

    // See comment in GroupedForceAndPot for explanation
    int n_write = blockDim.x / num;
    int a_write = threadIdx.x % num;   
    int read_b_offset = threadIdx.x / num;
    int valid = threadIdx.x < num * n_write;

    PosMass<3,float> xm_a = posm[prange.x + a_write];
    LocalExp<3,float> gloc_a = gloc[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    PosMass<3,float> gxm_a = {0.f,0.f,0.f,0.f};
    PosMass<3,float> gxm_a_kahan = {0.f,0.f,0.f,0.f};

    extern __shared__ PosMass<3,float> xm_b[];
    LocalExp<3,float>* gloc_b = (LocalExp<3,float>*) &xm_b[blockDim.x];

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();
        
        if(id >= 0) {
            xm_b[threadIdx.x] = posm[id];
            gloc_b[threadIdx.x] = gloc[id];
        }
        __syncthreads();

        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            PosMass<3,float> gxm_inc = VJP_GFPhiToGXM(xm_a, xm_b[ib], gloc_a, gloc_b[ib], softening2);
            kahan_add_vec(gxm_a.asvec, gxm_inc.asvec, gxm_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    PosMass<3,float>* gxm_shared = &xm_b[0];
    gxm_shared[threadIdx.x] = gxm_a;
    __syncthreads();

    if(read_b_offset == 0) {
        PosMass<3,float> gxm_cum = {0.f,0.f,0.f,0.f};
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(gxm_cum.asvec, gxm_shared[i*num + a_write].asvec, gxm_a_kahan.asvec);

        gposm_out[prange.x + a_write] = gxm_cum;
    }
}

#endif