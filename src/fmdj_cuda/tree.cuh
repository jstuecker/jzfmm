#ifndef CUSTOM_JAX_TREE_H
#define CUSTOM_JAX_TREE_H

#include <cub/cub.cuh>
#include <math_constants.h>

#include "common/data.cuh"
#include "common/math.cuh"

#if !defined(CUB_VERSION) || CUB_MAJOR_VERSION < 2
#error "CUB version 2.0.0 or higher required"
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                           Zorder Sort                                          */
/* ---------------------------------------------------------------------------------------------- */

// Wrapper of the z-order comparison function to use with CUB
struct PosIdLess {
    __device__ __forceinline__
    bool operator()(const PosId &a, const PosId &b) {
        return z_pos_less(a.pos, b.pos);
    }
};

// Prepare keys and ids for sorting
__global__ void PosKeyArangeKernel(const float3* pos_in, PosId *keyid_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keyid_out[idx].pos = pos_in[idx];
        keyid_out[idx].id = idx;
    }
}

std::string PosZorderSort(
    cudaStream_t stream, 
    const float3* pos_in, 
    PosId* pos_id_out,
    int* tmp_buffer,
    size_t size,
    size_t tmp_bytes,
    size_t block_size
) {
    // Initialize indices 0, 1, 2, ..., size-1
    PosKeyArangeKernel<<< div_ceil(size, block_size), block_size, 0, stream>>>(pos_in, pos_id_out, size);

    // We have an annoying problem here:
    // CUB requires a temporary storage buffer and it will usually tell us dynamically what the
    // size of it is (On first call with zero pointer).
    // Unfortunately, we may not allocate storage dynamically in an FFI call
    // Therefore, we have to estimate the storage requirements in advance in python/jax and pass
    // a sufficiently large buffer to the function (via "tmp_buffer").
    // Empirically I have found that the storage tends to be a bit larger than n * sizeof(PosId)
    // that is why we will pre-allocate something that is a few percent larger than that.
    // However, below we throw an error if our assumption ever turns out wrong.

    // find out the required storage size
    size_t required_storage_bytes;
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(nullptr, required_storage_bytes, pos_id_out, size, PosIdLess());
    
    // Check if the provided buffer is large enough
    if (tmp_bytes < required_storage_bytes) {
        return std::string(
            "The buffer in ZorderSort is too small. Please contact me if this check fails.") +
            std::string(" Have: ") + std::to_string(tmp_bytes) +
            std::string(". Required: ") + std::to_string(required_storage_bytes) +
            std::string(". Diff: ") + std::to_string((long long)required_storage_bytes - (long long)tmp_bytes);
    }
    
    // This is how we would allocate if we could. (Note that doing this breaks jit in some cases!)
    // cudaMallocAsync(&d_temp_storage, required_storage_bytes, stream); 

    // Run the sort
    cub::DeviceMergeSort::SortKeys<PosId*, int64_t, PosIdLess>(tmp_buffer, required_storage_bytes, pos_id_out, size, PosIdLess(), stream);
    
    return std::string();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         SummarizeLeaves                                        */
/* ---------------------------------------------------------------------------------------------- */

struct PosN {
    float3 pos;
    int32_t n;
};

__global__ void SummarizeLeaves(
    const PosN* xnleaf,
    const int* nleaves_filled,
    int32_t* split_flags,
    int max_size,
    int n_leaves,
    int scan_size
) {
    int nfilled = nleaves_filled[0];

    // Finds splitting points where the group of particles between each splitting point
    // can be summarized into a single leaf node that represents <= max_size particles
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data preceding and following our block into shared memory
    int nload = blockDim.x + 2*scan_size + 1;
    extern __shared__ unsigned char smem[];
    PosN*   xn = reinterpret_cast<PosN*>(smem);
    int32_t* level = reinterpret_cast<int32_t*>(xn + nload);

    int ioff = blockIdx.x * blockDim.x - scan_size - 1;
    
    // Note: we may load some points duplicate at the boundary, but that is ok (they will have 
    // level 0). Keeping it this way simplifies the indexing logic later
    for(int i = threadIdx.x; i < nload; i += blockDim.x) {
        int ifrom = ioff + i;
        if(ifrom < 0)
            xn[i] = {make_float3(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F), 0};
        else if(ifrom >= nfilled)
            xn[i] = {make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F), 0};
        else
            xn[i] = xnleaf[ifrom];
    }

    __syncthreads();
    for(int i = threadIdx.x; i < nload-1; i += blockDim.x) {
        level[i] = msb_diff_level(xn[i].pos, xn[i + 1].pos);
    }
    __syncthreads();

    // Find the boundaries of each node
    int idx = threadIdx.x + scan_size;
    int mylevel = level[idx];
    int lsize = 0;
    for(int i = idx - scan_size; i < idx; i++) {
        lsize = xn[i+1].n + (level[i] >= mylevel ? 0 : lsize);
    }
    int rsize = 0;
    for(int i = idx + scan_size; i > idx; i--) {
        rsize = xn[i].n + (level[i] >= mylevel ? 0 : rsize);
    }

    // Each maximum size node that is <= max_size is bounded by nodes that are > max_size
    // Therefore, we can find their splitting points by simply flagging all nodes that are > max_size
    bool is_split = lsize + rsize > max_size;

    // Additionally we set the beginning and end points to be splits
    is_split |= node_idx == 0; 
    is_split |= node_idx == nfilled;
    is_split &= node_idx <= nfilled;

    if (node_idx <= n_leaves) {
        split_flags[node_idx] = is_split ? mylevel : -1000;
    }
}

#endif // CUSTOM_JAX_TREE_H

/* ---------------------------------------------------------------------------------------------- */
/*                                          Tree Building                                         */
/* ---------------------------------------------------------------------------------------------- */

struct NodePointers {
    int32_t* levels;
    int32_t* lbound;
    int32_t* rbound;
    int32_t* lchild;
    int32_t* rchild;
};

__global__ void BinarySearchParents(const float3* pos_in, NodePointers nodes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid_thread = (idx < n-1);

    // Node indices are offset by 1, because we put a fake node at the beginning
    // but we don't launch the kernel for it
    int node = idx + 1; 

    int target_level, lvl_left, lvl_right;
    int lbound, rbound;
    float3 p1, p2;

    if (valid_thread) {
        // Calculate the level difference of our considered set of two points (=node)
        p1 = pos_in[idx];
        p2 = pos_in[idx + 1];

        target_level = msb_diff_level(p1, p2);

        // We do a binary search, trying to find the closest point to the left
        // that has a level difference of at least `level`
        int imin = -1, imax = idx+1;
        lvl_left = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
            lvl_left = msb_diff_level(pos_in[itest], p2);
            if (lvl_left > target_level) {
                imin = itest;
            } else {
                imax = itest;
            }
        }

        // Our array has two fake nodes at the beginning and end
        // that's why we have to offset the indices by 1
        lbound = imin+1;

        if(imin >= 0)
            lvl_left = msb_diff_level(p1, pos_in[imin]);
        else
            lvl_left = 388;
    }
    
    __syncthreads(); // Synchronize to reduce thread divergence

    if (valid_thread) {
        // Now find the right side parent
        int imin = idx, imax = n;
        lvl_right = 388;
        while (imin+1 < imax) {
            int itest = (imin + imax) / 2;
            lvl_right = msb_diff_level(p1, pos_in[itest]);
            if (lvl_right > target_level) {
                imax = itest;
            } else {
                imin = itest;
            }
        }

        rbound = imin+1;

        if(rbound < n)
            lvl_right = msb_diff_level(p1, pos_in[rbound]);
        else
            lvl_right = 388;
    }

    __syncthreads();

    if (valid_thread) {
        nodes.levels[node] = target_level;
        nodes.lbound[node] = lbound;
        nodes.rbound[node] = rbound;
        
        // The parent of each node is the lower one of the two boundary nodes
        if(lvl_left <= lvl_right) {
            nodes.rchild[lbound] = node;
        } else {
            nodes.lchild[rbound] = node;
        }
    }
}
__global__ void InitNodes(NodePointers nodes, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nnodes = n + 1;

    if (idx >= Nnodes)
        return;

    // We have fake nodes at the beginning and end of the array
    // to simplify walking the tree
    if((idx == 0) || (idx == Nnodes - 1)) {
        nodes.lbound[idx] = 0;
        nodes.rbound[idx] = Nnodes-1;
        nodes.levels[idx] = 388;
        nodes.lchild[idx] = idx; // Point to itself, may be overwritten later
        nodes.rchild[idx] = idx; // Point to itself, may be overwritten later
    }
    else {
        // indices <= 0 correspond to leafs (=particles)
        // by defaults node's children point to particles,
        // but about half of them will be overwritten by nodes later
        nodes.lchild[idx] = -idx + 1;
        nodes.rchild[idx] = -idx;
    }
}

void ZTreeNodeRelations(
    cudaStream_t stream, 
    const float3* pos_in,
    int* outputs,
    const size_t nleaves,
    const size_t block_size
) {
    // size_t n = pos_in.element_count()/3;
    size_t Nnodes = nleaves + 1;

    // Output will be (5, Nnodes) array with different types of information in the first axis
    // Create some easier readable pointers that start at offset locations in the output
    NodePointers nodes;
    nodes.levels = outputs;
    nodes.lbound = outputs + Nnodes;
    nodes.rbound = outputs + 2 * Nnodes;
    nodes.lchild = outputs + 3 * Nnodes;
    nodes.rchild = outputs + 4 * Nnodes;
    
    InitNodes<<< div_ceil(Nnodes, block_size), block_size, 0, stream>>>(nodes, nleaves);

    BinarySearchParents<<< div_ceil(nleaves-1, block_size), block_size, 0, stream>>>(pos_in, nodes, nleaves);
}