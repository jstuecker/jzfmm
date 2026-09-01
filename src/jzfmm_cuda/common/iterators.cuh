#ifndef COMMON_ITERATORS_CUH
#define COMMON_ITERATORS_CUH

struct SegmentManager {
    // Traverses a list of pointer-based segments as a block-wide linear stream.
    int istart, iend;
    const int* __restrict__ inodes;
    const int* __restrict__ isplits;
    int2* __restrict__ segments;
    int seg_loaded;

    int next_seg = 0;
    int seg_offset = 0;
    int num_loaded = 0;
    int nload_max = 0;

    __device__ __forceinline__ SegmentManager(
        const int* inodes_, const int* isplits_, int2* segments_,
        int istart_, int iend_, int nload_max_
    ) : istart(istart_), iend(iend_), inodes(inodes_), isplits(isplits_),
        segments(segments_), nload_max(nload_max_) {
        load_segments();
    }

    __device__ __forceinline__ void load_segments() {
        __syncthreads();
        seg_loaded = min(nload_max, iend - istart);
        if(threadIdx.x < seg_loaded) {
            int segment_idx = inodes ? inodes[istart + threadIdx.x] : istart + threadIdx.x;
            int i0 = isplits[segment_idx];
            int i1 = isplits[segment_idx + 1];
            segments[threadIdx.x] = {i0, i1 - i0};
        }
        istart += seg_loaded;
        next_seg = 0;
        __syncthreads();
    }

    __device__ __forceinline__ int next() {
        int id = -1;
        num_loaded = 0;
        while(!finished()) {
            if(next_seg >= seg_loaded)
                load_segments();

            int2 seg = segments[next_seg];
            int nadd = min(seg.y - seg_offset, blockDim.x - num_loaded);
            if(threadIdx.x >= num_loaded && threadIdx.x < num_loaded + nadd)
                id = seg.x + seg_offset + threadIdx.x - num_loaded;
            num_loaded += nadd;
            if(num_loaded >= blockDim.x) {
                seg_offset += nadd;
                break;
            }
            seg_offset = 0;
            next_seg += 1;
        }
        return id;
    }

    __device__ __forceinline__ bool finished() const {
        return next_seg >= seg_loaded && istart >= iend;
    }

    __device__ __forceinline__ int nids_loaded() const {
        return num_loaded;
    }
};

#endif // COMMON_ITERATORS_CUH
