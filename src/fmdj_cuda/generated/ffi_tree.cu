// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../common/math.cuh"
#include "../tree.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: PosZorderSort                             */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error PosZorderSortFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::Result<ffi::AnyBuffer> pos_id_out,
    ffi::Result<ffi::AnyBuffer> tmp_buffer,
    size_t block_size
) {
    size_t size = pos_in.element_count()/3;
    size_t tmp_bytes = tmp_buffer->size_bytes();

    // Now call our function
    std::string result = PosZorderSort(
        stream,
        reinterpret_cast<float3*>(pos_in.untyped_data()),
        reinterpret_cast<PosId*>(pos_id_out->untyped_data()),
        reinterpret_cast<int*>(tmp_buffer->untyped_data()),
        size,
        tmp_bytes,
        block_size
    );
    // Check if the function returned an error string
    if (!result.empty()) {
        return ffi::Error::Internal(result);
    }

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PosZorderSortFFI, PosZorderSortFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_in
        .Ret<ffi::AnyBuffer>() // pos_id_out
        .Ret<ffi::AnyBuffer>() // tmp_buffer
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SummarizeLeaves                           */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SummarizeLeavesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xnleaf,
    ffi::AnyBuffer nleaves_filled,
    ffi::Result<ffi::AnyBuffer> split_flags,
    int max_size,
    int scan_size,
    size_t block_size
) {
    int n_leaves = xnleaf.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(n_leaves+1, block_size));
    size_t smem = (block_size + 2*scan_size + 1) * (sizeof(PosN) + sizeof(int32_t));
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PosN* xnleaf_val = reinterpret_cast<PosN*>(xnleaf.untyped_data());
    int* nleaves_filled_val = reinterpret_cast<int*>(nleaves_filled.untyped_data());
    int32_t* split_flags_val = reinterpret_cast<int32_t*>(split_flags->untyped_data());

    void* args[] = {
        &xnleaf_val,
        &nleaves_filled_val,
        &split_flags_val,
        &max_size,
        &n_leaves,
        &scan_size
    };
    cudaLaunchKernel((const void*)SummarizeLeaves, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SummarizeLeavesFFI, SummarizeLeavesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xnleaf
        .Arg<ffi::AnyBuffer>() // nleaves_filled
        .Ret<ffi::AnyBuffer>() // split_flags
        .Attr<int>("max_size")
        .Attr<int>("scan_size")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: FindNodeBoundaries                        */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error FindNodeBoundariesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos_in,
    ffi::AnyBuffer nleaves,
    ffi::Result<ffi::AnyBuffer> nodes_levels,
    ffi::Result<ffi::AnyBuffer> nodes_lbound,
    ffi::Result<ffi::AnyBuffer> nodes_rbound,
    size_t block_size
) {
    int size_nodes = nodes_levels->element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* pos_in_val = reinterpret_cast<float3*>(pos_in.untyped_data());
    int* nleaves_val = reinterpret_cast<int*>(nleaves.untyped_data());
    int32_t* nodes_levels_val = reinterpret_cast<int32_t*>(nodes_levels->untyped_data());
    int32_t* nodes_lbound_val = reinterpret_cast<int32_t*>(nodes_lbound->untyped_data());
    int32_t* nodes_rbound_val = reinterpret_cast<int32_t*>(nodes_rbound->untyped_data());

    void* args[] = {
        &pos_in_val,
        &nleaves_val,
        &nodes_levels_val,
        &nodes_lbound_val,
        &nodes_rbound_val,
        &size_nodes
    };
    cudaLaunchKernel((const void*)FindNodeBoundaries, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FindNodeBoundariesFFI, FindNodeBoundariesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos_in
        .Arg<ffi::AnyBuffer>() // nleaves
        .Ret<ffi::AnyBuffer>() // nodes_levels
        .Ret<ffi::AnyBuffer>() // nodes_lbound
        .Ret<ffi::AnyBuffer>() // nodes_rbound
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GetNodeGeometry                           */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error GetNodeGeometryFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer lbound,
    ffi::AnyBuffer rbound,
    ffi::AnyBuffer nnodes,
    ffi::Result<ffi::AnyBuffer> level,
    ffi::Result<ffi::AnyBuffer> center,
    ffi::Result<ffi::AnyBuffer> extent,
    size_t block_size
) {
    int size_nodes = level->element_count();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(size_nodes, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float3* pos_val = reinterpret_cast<float3*>(pos.untyped_data());
    int* lbound_val = reinterpret_cast<int*>(lbound.untyped_data());
    int* rbound_val = reinterpret_cast<int*>(rbound.untyped_data());
    int* nnodes_val = reinterpret_cast<int*>(nnodes.untyped_data());
    int32_t* level_val = reinterpret_cast<int32_t*>(level->untyped_data());
    float3* center_val = reinterpret_cast<float3*>(center->untyped_data());
    float3* extent_val = reinterpret_cast<float3*>(extent->untyped_data());

    void* args[] = {
        &pos_val,
        &lbound_val,
        &rbound_val,
        &nnodes_val,
        &level_val,
        &center_val,
        &extent_val,
        &size_nodes
    };
    cudaLaunchKernel((const void*)GetNodeGeometry, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GetNodeGeometryFFI, GetNodeGeometryFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // lbound
        .Arg<ffi::AnyBuffer>() // rbound
        .Arg<ffi::AnyBuffer>() // nnodes
        .Ret<ffi::AnyBuffer>() // level
        .Ret<ffi::AnyBuffer>() // center
        .Ret<ffi::AnyBuffer>() // extent
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_tree, m) {
    m.def("PosZorderSort", []() { return EncapsulateFfiCall(&PosZorderSortFFI); });
    m.def("SummarizeLeaves", []() { return EncapsulateFfiCall(&SummarizeLeavesFFI); });
    m.def("FindNodeBoundaries", []() { return EncapsulateFfiCall(&FindNodeBoundariesFFI); });
    m.def("GetNodeGeometry", []() { return EncapsulateFfiCall(&GetNodeGeometryFFI); });
}