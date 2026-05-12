#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int dim>
__forceinline__ __device__ LocalExp<dim,float> zero_local_exp() {
    LocalExp<dim,float> loc;
    loc.pot = 0.f;
    loc.grad = Vec<dim,float>::constant(0.f);
    return loc;
}

template<int dim>
__forceinline__ __device__ PosMass<dim,float> zero_pos_mass() {
    PosMass<dim,float> xm;
    xm.pos = Vec<dim,float>::constant(0.f);
    xm.mass = 0.f;
    return xm;
}

template<int dim>
__forceinline__ __device__ LocalExp<dim,float> GetForceAndPot(
    PosMass<dim,float> xmi,
    PosMass<dim,float> xmj,
    float softening2
) {
    Vec<dim,float> dx = xmj.pos - xmi.pos;
    float rinv = rsqrtf(dx.norm2() + softening2);
    float minvr = -xmj.mass * rinv;

    LocalExp<dim,float> loc;
    loc.pot = minvr;
    loc.grad = (minvr * rinv * rinv) * dx;

    return loc;
}

template<int dim>
__forceinline__ __device__ PosMass<dim,float> VJP_GFPhiToGXM(
    const PosMass<dim,float> xmi, const PosMass<dim,float> xmj,
    const LocalExp<dim,float> gi, const LocalExp<dim,float> gj,
    float epsilon2
) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    Vec<dim,float> dx = xmj.pos - xmi.pos;
    float r2 = dx.norm2();
    float rinv = r2 > 1e-10f * epsilon2 ? rsqrtf(r2 + epsilon2) : 0.f;
    float rinv2 = rinv*rinv;

    float f1 = -rinv*rinv2;
    float f2 = 3*rinv*rinv2*rinv2;

    Vec<dim,float> gm_diff = xmi.mass * gj.grad - xmj.mass * gi.grad;
    float fgdiff = f2*gm_diff.dot(dx) + f1 * (gi.pot * xmj.mass + gj.pot * xmi.mass);

    PosMass<dim,float> gxmi;

    gxmi.pos = f1 * gm_diff + fgdiff * dx;
    gxmi.mass = - f1 * dx.dot(gj.grad) - rinv * gj.pot;

    return gxmi;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan, int dim>
__global__ void ForceAndPotential(
    const PosMass<dim,float> *xm,
    LocalExp<dim,float> *loc_out,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass<dim,float> xmi = zero_pos_mass<dim>();
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ unsigned char force_smem[];
    PosMass<dim,float>* xmj_shared = reinterpret_cast<PosMass<dim,float>*>(force_smem);

    LocalExp<dim,float> loc_i = zero_local_exp<dim>();
    LocalExp<dim,float> loc_kahan = zero_local_exp<dim>();

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            LocalExp<dim,float> loc_new = GetForceAndPot<dim>(xmi, xmj_shared[j], epsilon2);
            kahan_add_vec(loc_i.asvec, loc_new.asvec, loc_kahan.asvec);
        }
    }

    loc_i.pot += xmi.mass / epsilon; // remove self-interaction from potential

    if(ipart < n)
        loc_out[blockIdx.x * blockDim.x + threadIdx.x] = loc_i;
}

template <bool kahan, int dim>
__global__ void BwdForceAndPotential(
    const LocalExp<dim,float> *gloc,
    const PosMass<dim,float> *xm,
    PosMass<dim,float> *gxm,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    PosMass<dim,float> xmi = zero_pos_mass<dim>();
    LocalExp<dim,float> gloc_i = zero_local_exp<dim>();

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    if(ipart < n) {
        xmi = xm[ipart];
        gloc_i = gloc[ipart];
    }

    extern __shared__ unsigned char bwd_force_smem[];
    PosMass<dim,float>* xmj_shared = reinterpret_cast<PosMass<dim,float>*>(bwd_force_smem);
    LocalExp<dim,float>* gloc_j_shared = reinterpret_cast<LocalExp<dim,float>*>(
        bwd_force_smem + blockDim.x * sizeof(PosMass<dim,float>)
    );

    PosMass<dim,float> gxm_i = zero_pos_mass<dim>();
    PosMass<dim,float> gxm_i_kahan = zero_pos_mass<dim>();

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num) {
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
            gloc_j_shared[threadIdx.x] = gloc[jblock * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        for (int j = 0; j < num; j++) {
            PosMass<dim,float> gxm_inc = VJP_GFPhiToGXM<dim>(
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

template <bool kahan, int dim>
__global__ void GroupedForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PosMass<dim,float>* posm,
    // outputs:
    LocalExp<dim,float>* loc_out,
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

    PosMass<dim,float> xaWrite = posm[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    LocalExp<dim,float> loc_a = zero_local_exp<dim>();
    LocalExp<dim,float> loc_a_kahan = zero_local_exp<dim>();

    extern __shared__ unsigned char grouped_force_smem[];
    PosMass<dim,float>* xm_b = reinterpret_cast<PosMass<dim,float>*>(grouped_force_smem);

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0)
            xm_b[threadIdx.x] = posm[id];
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            LocalExp<dim,float> loc_new = GetForceAndPot<dim>(xaWrite, xm_b[ib], softening2);
            add_vec<kahan>(loc_a.asvec, loc_new.asvec, loc_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    LocalExp<dim,float>* loc_shared = reinterpret_cast<LocalExp<dim,float>*>(grouped_force_smem);
    loc_shared[threadIdx.x] = loc_a;
    __syncthreads();

    if(read_b_offset == 0) {
        LocalExp<dim,float> loc_cum = zero_local_exp<dim>();
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(loc_cum.asvec, loc_shared[i*num + a_write].asvec, loc_a_kahan.asvec);

        if(valid)
            loc_out[prange.x + a_write] = loc_cum;
    }
}

template <bool kahan, int dim>
__global__ void BwdGroupedForceAndPot(
    // inputs:
    const int2* node_range,
    const int* spl_nodes,
    const int* spl_ilist,
    const int* ilist_nodes,
    const PosMass<dim,float>* posm,
    const LocalExp<dim,float> *gloc,
    // outputs:
    PosMass<dim,float>* gposm_out,
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

    PosMass<dim,float> xm_a = posm[prange.x + a_write];
    LocalExp<dim,float> gloc_a = gloc[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_nodes,
        spl_nodes,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    PosMass<dim,float> gxm_a = zero_pos_mass<dim>();
    PosMass<dim,float> gxm_a_kahan = zero_pos_mass<dim>();

    extern __shared__ unsigned char bwd_grouped_force_smem[];
    PosMass<dim,float>* xm_b = reinterpret_cast<PosMass<dim,float>*>(bwd_grouped_force_smem);
    LocalExp<dim,float>* gloc_b = reinterpret_cast<LocalExp<dim,float>*>(
        bwd_grouped_force_smem + blockDim.x * sizeof(PosMass<dim,float>)
    );

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();
        
        if(id >= 0) {
            xm_b[threadIdx.x] = posm[id];
            gloc_b[threadIdx.x] = gloc[id];
        }
        __syncthreads();

        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            PosMass<dim,float> gxm_inc = VJP_GFPhiToGXM<dim>(xm_a, xm_b[ib], gloc_a, gloc_b[ib], softening2);
            kahan_add_vec(gxm_a.asvec, gxm_inc.asvec, gxm_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    PosMass<dim,float>* gxm_shared = reinterpret_cast<PosMass<dim,float>*>(bwd_grouped_force_smem);
    gxm_shared[threadIdx.x] = gxm_a;
    __syncthreads();

    if(read_b_offset == 0) {
        PosMass<dim,float> gxm_cum = zero_pos_mass<dim>();
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(gxm_cum.asvec, gxm_shared[i*num + a_write].asvec, gxm_a_kahan.asvec);

        gposm_out[prange.x + a_write] = gxm_cum;
    }
}

#endif
