#ifndef FMM_H
#define FMM_H

#include "multipoles.cuh"
#include "common/math.cuh"
#include "common/data.cuh"
#include "common/iterators.cuh"
#include "opening.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                     CountInteractionsAndM2L                                    */
/* ---------------------------------------------------------------------------------------------- */

// Todo:
// I think I can still optimize the input/output pattern of the kernels below 
// and drastically reduce the amount of data that is needed to store the interaction list
// This may be done by summarzing the interaction list, wherever nodes are continguous
// e.g. 4,5,6,7,10,11,12,14,16 -> 4-7 10-12 14 16 ...
// This will also allow to optimally coalesce memory access and reduce the overhead that
// comes with the indexing scheme.
// I have measured that ~72% of interactions are int[i+1] == int[i]+1
// 59% are int[i+2] == int[i]+2 (for these cases we can actually save memory)

#define BLOCKSIZE 32
#define MAX_NUMA 16

#define ALLTHREADS 0xFFFFFFFF

template<int opening_criterion_kind, int p, int dim, typename tvec>
__global__ void CountInteractionsAndM2L(
    // inputs:
    const int2* node_range,
    const int* spl_nodes_recv,
    const int* spl_nodes_src,
    const int* spl_ilist,
    const int* ilist_isrc,
    const Node<dim,tvec>* children_recv,
    const Node<dim,tvec>* children_src,
    const tvec* mp_src,
    const tvec* radial_kernel_params,
    const tvec* opening_criterion_params,
    const int* single_thread_per_receiver,
    const tvec* loc_in,
    // outputs:
    tvec* loc_recv,
    int* ilist_child_count_out,
    // attributes:
    int radial_kernel_kind
) {
    constexpr int ncomb_mp = NCOMB(p, dim);
    constexpr int ncomb_loc = NCOMB(p, dim);
    constexpr int m2l_blocksize =
        (p == 7 && std::is_same_v<tvec, double>) ? 16 : BLOCKSIZE;
    auto opening_criterion = OpeningCriterion<opening_criterion_kind>::template make_params<tvec>(opening_criterion_params);

    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 child_range = {spl_nodes_recv[nodeid], spl_nodes_recv[nodeid + 1]};

    // This loop should handle almost always everything on the first pass
    // However, to deal with edge cases we have to loop over scenarios where 
    // the node has more than MAX_NUMA children
    for(int offsetA=child_range.x; offsetA < child_range.y; offsetA += MAX_NUMA) {
        int num_childrenA = min(MAX_NUMA, child_range.y - offsetA);

        // childA info
        __shared__ Node<dim,tvec> childA[MAX_NUMA];
        __shared__ int scale_expA[MAX_NUMA];
        if(threadIdx.x < num_childrenA) {
            Node<dim,tvec> child = children_recv[offsetA + threadIdx.x];
            childA[threadIdx.x] = child;
            scale_expA[threadIdx.x] = lvl_vec<dim>(child.level)[dim - 1];
        }

        int num_open[MAX_NUMA];
        #pragma unroll
        for(int i = 0; i < MAX_NUMA; i++) {
            num_open[i] = 0;
        }

        Vec<ncomb_loc,tvec> LocA;
        #pragma unroll
        for(int i = 0; i < ncomb_loc; i++) {
            LocA[i] = tvec(0);
        }

        // Child B info. This is transposed to reduce smem bank conflicts
        // Todo: BLOCKSIZE does not need to be a compile time constant here.
        //       Make it more flexible! (Need to adapt the warp communication scheme below though!)
        __shared__ Vec<dim,tvec> posB[m2l_blocksize];
        __shared__ int scale_expB[m2l_blocksize];
        __shared__ Vec<ncomb_mp,tvec> mpB[m2l_blocksize];
        
        // Interaction list info:
        int2 ilist_range = {spl_ilist[nodeid], spl_ilist[nodeid + 1]};
        __syncthreads();

        __shared__ int2 segments[m2l_blocksize];
        SegmentManager seg_mgr(
            ilist_isrc,
            spl_nodes_src,
            segments,
            ilist_range.x,
            ilist_range.y,
            m2l_blocksize
        );

        // Todo:
        // I realized that it is better to discard the residual threads, as is done in the
        // force kernel. Also do that here!

        // Precalculate layout for M2L interactions
        bool single_thread = single_thread_per_receiver[0] != 0;
        bool valid_thread;
        int a_write;
        int read_b_offset;
        int residual_threads = blockDim.x % num_childrenA;
        int n_write_a;

        if(single_thread) {
            // Deterministic top-level mode: all threads still serve as source lanes,
            // but only one thread accumulates for each receiver child.
            valid_thread = threadIdx.x < num_childrenA;
            a_write = threadIdx.x;
            read_b_offset = 0;
            n_write_a = 1;
        }
        else {
            // Normal mode: split the source interactions for each receiver across the block.
            valid_thread = true;
            a_write = threadIdx.x % num_childrenA;
            read_b_offset = threadIdx.x / num_childrenA;
            n_write_a = blockDim.x / num_childrenA + (a_write < residual_threads);
        }

        Vec<dim,tvec> xaWrite = childA[valid_thread ? a_write : 0].center;

        while(!seg_mgr.finished()) {
            int id = seg_mgr.next();

            // Each thread loads one other child B to check the opening criterion
            Node<dim,tvec> childB;
            if(id >= 0) {
                childB = children_src[id];
            }

            // For each child A, we count the cumulative number of opens and we 
            // flag the M2L interactions that need to be evaluated now

            unsigned int interact_flags_wa = 0;

            bool any_interacts = false;
            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(i >= num_childrenA)
                    continue;

                bool need_open = (id >= 0) && OpeningCriterion<opening_criterion_kind>::template should_open<dim,tvec>(
                    childA[i], childB, opening_criterion
                );
                bool actually_open = need_open && (id >= 0);
                bool interact_now = !need_open && (id >= 0);
                any_interacts = any_interacts || interact_now;

                // Sum over all threads
                num_open[i] += __popc(__ballot_sync(__activemask(), actually_open));
                // Flag the active m2l interactions for this child
                unsigned int interact_flags = __ballot_sync(__activemask(), interact_now);
                // we only store the flag for the child that we need to write to later
                interact_flags_wa = (valid_thread && (i == a_write)) ? interact_flags : interact_flags_wa;
            }

            __syncthreads();

            // only read the multipoles if at least one interaction happens with this childB
            if(any_interacts) {
                posB[threadIdx.x] = childB.center;
                scale_expB[threadIdx.x] =
                    lvl_vec<dim>(children_src[id].level)[dim - 1];
                for(int k=0; k<ncomb_mp; k++) {
                    mpB[threadIdx.x][k] = mp_src[id * ncomb_mp + k];
                }
            }

            __syncthreads();

            // Now we have all the data we need for the M2L interactions in shared memory.
            // To avoid reduction operations across threads, we transpose the problem differently here.

            if(valid_thread) {
                int ninteractionsB_withA = __popc(interact_flags_wa);
                for(int ib=read_b_offset; ib < ninteractionsB_withA; ib += n_write_a) {
                    // have to add the m2l interactions between a_write and the ib-th set bit in 
                    // interact_flags_wa

                    // find the ib-th set bit
                    int b_read = __fns(interact_flags_wa, 0, ib+1);

                    m2l_translator<p,dim,tvec>(
                        posB[b_read] - xaWrite,
                        scale_expB[b_read], scale_expA[a_write],
                        mpB[b_read], LocA,
                        radial_kernel_kind, radial_kernel_params
                    );
                }
            }
        }

        __syncthreads();
        if(threadIdx.x == 0) { 
            // Since num_open lives in registers, the simplest way of writing it is with a single thread
            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(offsetA + i < child_range.y) {
                    ilist_child_count_out[offsetA + i] = num_open[i];
                }
            }
        }

        if(single_thread) {
            if(threadIdx.x < num_childrenA) {
                #pragma unroll
                for(int i = 0; i < ncomb_loc; i++) {
                    int iout = (offsetA + threadIdx.x) * ncomb_loc + i;
                    loc_recv[iout] = loc_in[iout] + LocA[i];
                }
            }
            __syncthreads();
            continue;
        }

        // Now we need to reduce the local terms accross threads
        // with the same output particle.
        __shared__ Vec<ncomb_loc,tvec> loc_partials[m2l_blocksize];
        loc_partials[threadIdx.x] = LocA;
        __syncthreads();

        if(threadIdx.x < num_childrenA) {
            Vec<ncomb_loc,tvec> loc_sum;

            #pragma unroll
            for(int i = 0; i < ncomb_loc; i++) {
                loc_sum[i] = tvec(0);
            }

            int n_write_child = blockDim.x / num_childrenA + (threadIdx.x < residual_threads);
            for(int iw = 0; iw < n_write_child; iw++) {
                loc_sum += loc_partials[iw * num_childrenA + threadIdx.x];
            }

            #pragma unroll
            for(int i = 0; i < ncomb_loc; i++) {
                int iout = (offsetA + threadIdx.x) * ncomb_loc + i;
                loc_recv[iout] = loc_in[iout] + loc_sum[i];
            }
        }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       InsertInteractions                                       */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ int nbits_set_before(unsigned mask, int bit)
{
    unsigned lower_mask = (bit == 0) ? 0u : ((1u << bit) - 1u);
    return __popc(mask & lower_mask);
}

template<int opening_criterion_kind, int dim, typename tvec>
__global__ void InsertInteractions(
    // inputs:
    const int2* node_range,
    const int* spl_nodes_recv,
    const int* spl_nodes_src,
    const int* spl_ilist,
    const int* ilist_isrc,
    const Node<dim,tvec>* children_recv,
    const Node<dim,tvec>* children_src,
    const int* spl_ilist_child,
    const tvec* opening_criterion_params,
    // outputs:
    int* child_ilist_out
) {
    auto opening_criterion = OpeningCriterion<opening_criterion_kind>::template make_params<tvec>(opening_criterion_params);

    // Node A info:
    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y) {
        return;
    }

    int2 child_range = {spl_nodes_recv[nodeid], spl_nodes_recv[nodeid + 1]};

    // This loop should handle almost always everything on the first pass
    // However, to deal with edge cases we have to loop over scenarios where 
    // the node has more than MAX_NUMA children
    for(int offsetA=child_range.x; offsetA < child_range.y; offsetA += MAX_NUMA) {
        int num_childrenA = min(MAX_NUMA, child_range.y - offsetA);

        // childA info
        __shared__ Node<dim,tvec> childA[MAX_NUMA];
        __shared__ int ilist_offsets[MAX_NUMA];
        if(threadIdx.x < num_childrenA) {
            Node<dim,tvec> child = children_recv[offsetA + threadIdx.x];
            childA[threadIdx.x] = child;
            ilist_offsets[threadIdx.x] = spl_ilist_child[offsetA + threadIdx.x];
        }

        int num_open[MAX_NUMA];
        #pragma unroll
        for(int i = 0; i < MAX_NUMA; i++) {
            num_open[i] = 0;
        }

        // Interaction list info:
        int2 ilist_range = {spl_ilist[nodeid], spl_ilist[nodeid + 1]};
        __syncthreads();

        __shared__ int2 segments[BLOCKSIZE];
        SegmentManager seg_mgr(
            ilist_isrc,
            spl_nodes_src,
            segments,
            ilist_range.x,
            ilist_range.y,
            BLOCKSIZE
        );

        while(!seg_mgr.finished()) {
            int id = seg_mgr.next();

            // Each thread loads one other child B to check the opening criterion
            Node<dim,tvec> childB;
            if(id >= 0) {
                childB = children_src[id];
            }

            #pragma unroll
            for(int i = 0; i < MAX_NUMA; i++) {
                if(i >= num_childrenA)
                    continue;

                bool need_open = (id >= 0) && OpeningCriterion<opening_criterion_kind>::template should_open<dim,tvec>(
                    childA[i], childB, opening_criterion
                );

                unsigned open_mask = __ballot_sync(ALLTHREADS, need_open);

                // Count the number of activated bits before our thread's bit
                int warp_offset = nbits_set_before(open_mask, threadIdx.x & 0x1f);

                if(need_open)
                    child_ilist_out[ilist_offsets[i] + num_open[i] + warp_offset] = id;
                
                num_open[i] += __popc(open_mask);
            }
        }
    }
}


#endif
