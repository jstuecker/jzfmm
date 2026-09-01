#ifndef FORCES_CUH
#define FORCES_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "radial_kernels.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                        Helper Functions                                        */
/* ---------------------------------------------------------------------------------------------- */

template<int dim, typename tvec>
__forceinline__ __device__ LocalExp<dim,tvec> zero_local_exp() {
    LocalExp<dim,tvec> loc;
    loc.pot = tvec(0);
    loc.grad = Vec<dim,tvec>::constant(tvec(0));
    return loc;
}

template<int dim, typename tvec>
__forceinline__ __device__ PosMass<dim,tvec> zero_pos_mass() {
    PosMass<dim,tvec> xm;
    xm.pos = Vec<dim,tvec>::constant(tvec(0));
    xm.mass = tvec(0);
    return xm;
}

template<int radial_kernel_kind, int dim, typename tvec>
__forceinline__ __device__ LocalExp<dim,tvec> EvaluatePairInteraction(
    PosMass<dim,tvec> xmi,
    PosMass<dim,tvec> xmj,
    typename RadialKernel<radial_kernel_kind>::template Params<tvec> radial_kernel
) {
    Vec<dim,tvec> dx = xmj.pos - xmi.pos;
    tvec r2 = dx.norm2();

    Vec<2,tvec> coeffs;
    RadialKernel<radial_kernel_kind>::template r2_derivative_coeffs<1,tvec>(r2, radial_kernel, coeffs);

    LocalExp<dim,tvec> loc;
    loc.pot = xmj.mass * coeffs[0];
    loc.grad = -(xmj.mass * coeffs[1]) * dx;

    return loc;
}

template<int radial_kernel_kind, int dim, typename tvec>
__forceinline__ __device__ PosMass<dim,tvec> VJP_GFPhiToGXM(
    const PosMass<dim,tvec> xmi, const PosMass<dim,tvec> xmj,
    const LocalExp<dim,tvec> gi, const LocalExp<dim,tvec> gj,
    typename RadialKernel<radial_kernel_kind>::template Params<tvec> radial_kernel
) {
    // calculates the vector jacobian product of the interaction between xmi and xmj
    // gi and gj are the final gradient vectors of fphi_i and fphi_j, respectively.
    // we have to back propagate the gradient towards a gradient with respect to xmi
    // for understanding the maths, please consider the corresponding .ipynb notebook
    Vec<dim,tvec> dx = xmj.pos - xmi.pos;
    tvec r2 = dx.norm2();

    Vec<3,tvec> coeffs;
    RadialKernel<radial_kernel_kind>::template r2_derivative_coeffs<2,tvec>(r2, radial_kernel, coeffs);
    tvec K = r2 > tvec(1e-20) ? coeffs[0] : tvec(0);
    tvec f1 = r2 > tvec(1e-20) ? coeffs[1] : tvec(0);
    tvec f2 = r2 > tvec(1e-20) ? coeffs[2] : tvec(0);

    Vec<dim,tvec> gm_diff = xmi.mass * gj.grad - xmj.mass * gi.grad;
    tvec fgdiff = f2*gm_diff.dot(dx) + f1 * (gi.pot * xmj.mass + gj.pot * xmi.mass);

    PosMass<dim,tvec> gxmi;

    gxmi.pos = tvec(-1) * (f1 * gm_diff + fgdiff * dx);
    gxmi.mass = f1 * dx.dot(gj.grad) + K * gj.pot;

    return gxmi;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                  Direct Pair Summation Kernel                                  */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan, int radial_kernel_kind, int dim, typename tvec>
__global__ void DirectPairSummation(
    const PosMass<dim,tvec> *xm,
    const tvec* radial_kernel_params,
    LocalExp<dim,tvec> *loc_out,
    int n,
    const bool remove_self_interaction
) {
    const int steps = div_ceil(n, blockDim.x);
    auto radial_kernel = RadialKernel<radial_kernel_kind>::template make_params<tvec>(radial_kernel_params);

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass<dim,tvec> xmi = zero_pos_mass<dim,tvec>();
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ unsigned char force_smem[];
    PosMass<dim,tvec>* xmj_shared = reinterpret_cast<PosMass<dim,tvec>*>(force_smem);

    LocalExp<dim,tvec> loc_i = zero_local_exp<dim,tvec>();
    LocalExp<dim,tvec> loc_kahan = zero_local_exp<dim,tvec>();

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            LocalExp<dim,tvec> loc_new = EvaluatePairInteraction<radial_kernel_kind,dim,tvec>(xmi, xmj_shared[j], radial_kernel);
            kahan_add_vec(loc_i.asvec, loc_new.asvec, loc_kahan.asvec);
        }
    }

    if(remove_self_interaction)
        loc_i.pot -= xmi.mass * radial_kernel_value<radial_kernel_kind,tvec>(tvec(0), radial_kernel);

    if(ipart < n)
        loc_out[blockIdx.x * blockDim.x + threadIdx.x] = loc_i;
}

template <bool kahan, int radial_kernel_kind, int dim, typename tvec>
__global__ void BwdDirectPairSummation(
    const LocalExp<dim,tvec> *gloc,
    const PosMass<dim,tvec> *xm,
    const tvec* radial_kernel_params,
    PosMass<dim,tvec> *gxm,
    int n,
    const bool remove_self_interaction
) {
    const int steps = div_ceil(n, blockDim.x);
    auto radial_kernel = RadialKernel<radial_kernel_kind>::template make_params<tvec>(radial_kernel_params);

    PosMass<dim,tvec> xmi = zero_pos_mass<dim,tvec>();
    LocalExp<dim,tvec> gloc_i = zero_local_exp<dim,tvec>();

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    if(ipart < n) {
        xmi = xm[ipart];
        gloc_i = gloc[ipart];
    }

    extern __shared__ unsigned char bwd_force_smem[];
    PosMass<dim,tvec>* xmj_shared = reinterpret_cast<PosMass<dim,tvec>*>(bwd_force_smem);
    LocalExp<dim,tvec>* gloc_j_shared = reinterpret_cast<LocalExp<dim,tvec>*>(
        bwd_force_smem + blockDim.x * sizeof(PosMass<dim,tvec>)
    );

    PosMass<dim,tvec> gxm_i = zero_pos_mass<dim,tvec>();
    PosMass<dim,tvec> gxm_i_kahan = zero_pos_mass<dim,tvec>();

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num) {
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
            gloc_j_shared[threadIdx.x] = gloc[jblock * blockDim.x + threadIdx.x];
        }
        __syncthreads();

        for (int j = 0; j < num; j++) {
            PosMass<dim,tvec> gxm_inc = VJP_GFPhiToGXM<radial_kernel_kind,dim,tvec>(
                xmi, xmj_shared[j],
                gloc_i, gloc_j_shared[j],
                radial_kernel
            );
            kahan_add_vec(gxm_i.asvec, gxm_inc.asvec, gxm_i_kahan.asvec);
        }
    }

    if((!remove_self_interaction) && (ipart < n))
        gxm_i.mass += radial_kernel_value<radial_kernel_kind,tvec>(tvec(0), radial_kernel) * gloc_i.pot;

    if(ipart < n)
        gxm[ipart] = gxm_i;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                Leaf-Leaf Pair Summation Kernel                                 */
/* ---------------------------------------------------------------------------------------------- */

template <bool kahan, int radial_kernel_kind, int dim, typename tvec>
__global__ void LeafLeafPairSummation(
    // inputs:
    const int2* node_range,
    const int* spl_recv,
    const int* spl_src,
    const int* spl_ilist,
    const int* ilist_isrc,
    const PosMass<dim,tvec>* posm_recv,
    const PosMass<dim,tvec>* posm_src,
    const tvec* radial_kernel_params,
    const LocalExp<dim,tvec>* loc_in,
    // outputs:
    LocalExp<dim,tvec>* loc_recv,
    // attributes:
    const bool remove_self_interaction
) {
    auto radial_kernel = RadialKernel<radial_kernel_kind>::template make_params<tvec>(radial_kernel_params);

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_recv[nodeid], spl_recv[nodeid + 1]};

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

    PosMass<dim,tvec> xaWrite = posm_recv[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_isrc,
        spl_src,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    LocalExp<dim,tvec> loc_a = zero_local_exp<dim,tvec>();
    LocalExp<dim,tvec> loc_a_kahan = zero_local_exp<dim,tvec>();

    extern __shared__ unsigned char grouped_force_smem[];
    PosMass<dim,tvec>* xm_b = reinterpret_cast<PosMass<dim,tvec>*>(grouped_force_smem);

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        // Each thread loads one other particle B
        if(id >= 0)
            xm_b[threadIdx.x] = posm_src[id];
        __syncthreads();

        // Now compute interactions
        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            LocalExp<dim,tvec> loc_new = EvaluatePairInteraction<radial_kernel_kind,dim,tvec>(xaWrite, xm_b[ib], radial_kernel);
            add_vec<kahan>(loc_a.asvec, loc_new.asvec, loc_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    LocalExp<dim,tvec>* loc_shared = reinterpret_cast<LocalExp<dim,tvec>*>(grouped_force_smem);
    loc_shared[threadIdx.x] = loc_a;
    __syncthreads();

    if(read_b_offset == 0) {
        LocalExp<dim,tvec> loc_cum = zero_local_exp<dim,tvec>();
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(loc_cum.asvec, loc_shared[i*num + a_write].asvec, loc_a_kahan.asvec);

        if(valid) {
            loc_cum.asvec += loc_in[prange.x + a_write].asvec;
            if(remove_self_interaction)
                loc_cum.pot -= xaWrite.mass * radial_kernel_value<radial_kernel_kind,tvec>(tvec(0), radial_kernel);
            loc_recv[prange.x + a_write] = loc_cum;
        }
    }
}

template <bool kahan, int radial_kernel_kind, int dim, typename tvec>
__global__ void BwdLeafLeafPairSummation(
    // inputs:
    const int2* node_range,
    const int* spl_recv,
    const int* spl_src,
    const int* spl_ilist,
    const int* ilist_isrc,
    const PosMass<dim,tvec>* posm_recv,
    const PosMass<dim,tvec>* posm_src,
    const tvec* radial_kernel_params,
    const LocalExp<dim,tvec> *gloc_recv,
    const LocalExp<dim,tvec> *gloc_src,
    // outputs:
    PosMass<dim,tvec>* gposm_recv,
    // attributes:
    const bool remove_self_interaction
) {
    auto radial_kernel = RadialKernel<radial_kernel_kind>::template make_params<tvec>(radial_kernel_params);

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;
    
    int2 prange = {spl_recv[nodeid], spl_recv[nodeid + 1]};

    int num = prange.y-prange.x;

    // See comment in LeafLeafPairSummation for explanation
    int n_write = blockDim.x / num;
    int a_write = threadIdx.x % num;   
    int read_b_offset = threadIdx.x / num;
    int valid = threadIdx.x < num * n_write;

    PosMass<dim,tvec> xm_a = posm_recv[prange.x + a_write];
    LocalExp<dim,tvec> gloc_a = gloc_recv[prange.x + a_write];

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_isrc,
        spl_src,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    PosMass<dim,tvec> gxm_a = zero_pos_mass<dim,tvec>();
    PosMass<dim,tvec> gxm_a_kahan = zero_pos_mass<dim,tvec>();

    extern __shared__ unsigned char bwd_grouped_force_smem[];
    PosMass<dim,tvec>* xm_b = reinterpret_cast<PosMass<dim,tvec>*>(bwd_grouped_force_smem);
    LocalExp<dim,tvec>* gloc_b = reinterpret_cast<LocalExp<dim,tvec>*>(
        bwd_grouped_force_smem + blockDim.x * sizeof(PosMass<dim,tvec>)
    );

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();
        
        if(id >= 0) {
            xm_b[threadIdx.x] = posm_src[id];
            gloc_b[threadIdx.x] = gloc_src[id];
        }
        __syncthreads();

        for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
            PosMass<dim,tvec> gxm_inc = VJP_GFPhiToGXM<radial_kernel_kind,dim,tvec>(xm_a, xm_b[ib], gloc_a, gloc_b[ib], radial_kernel);
            kahan_add_vec(gxm_a.asvec, gxm_inc.asvec, gxm_a_kahan.asvec);
        }
        __syncthreads();
    }

    // Now sum over all contributions to the same write position in shared memory
    PosMass<dim,tvec>* gxm_shared = reinterpret_cast<PosMass<dim,tvec>*>(bwd_grouped_force_smem);
    gxm_shared[threadIdx.x] = gxm_a;
    __syncthreads();

    if(read_b_offset == 0) {
        PosMass<dim,tvec> gxm_cum = zero_pos_mass<dim,tvec>();
        
        for(int i=0; i < n_write; i++)
            kahan_add_vec(gxm_cum.asvec, gxm_shared[i*num + a_write].asvec, gxm_a_kahan.asvec);

        if(!remove_self_interaction)
            gxm_cum.mass += radial_kernel_value<radial_kernel_kind,tvec>(tvec(0), radial_kernel) * gloc_a.pot;

        gposm_recv[prange.x + a_write] = gxm_cum;
    }
}

#endif
