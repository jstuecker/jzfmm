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
#include "../pair_summation.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: DirectPairSummation                       */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error DirectPairSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::AnyBuffer radial_kernel_params,
    ffi::Result<ffi::AnyBuffer> loc_out,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int n = xm.dimensions()[0];
    int dim = xm.dimensions()[1] - 1;
    DT tvec = xm.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(xm.dimensions()[0], block_size));
    size_t smem = blockDim.x * (dim + 1) * (xm.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* xm_arg = xm.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* loc_out_arg = loc_out->untyped_data();
    void* args[] = {
        &xm_arg,
        &radial_kernel_params_arg,
        &loc_out_arg,
        &n
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 0, 3, float>) },
        { {true, 0, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 0, 4, float>) },
        { {true, 0, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 0, 5, float>) },
        { {true, 0, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 0, 6, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 1, 3, float>) },
        { {true, 1, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 1, 4, float>) },
        { {true, 1, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 1, 5, float>) },
        { {true, 1, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 1, 6, float>) },
        { {true, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 2, 2, float>) },
        { {true, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 2, 3, float>) },
        { {true, 2, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 2, 4, float>) },
        { {true, 2, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 2, 5, float>) },
        { {true, 2, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 2, 6, float>) },
        { {true, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 3, 2, float>) },
        { {true, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 3, 3, float>) },
        { {true, 3, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 3, 4, float>) },
        { {true, 3, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 3, 5, float>) },
        { {true, 3, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<true, 3, 6, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 0, 3, float>) },
        { {false, 0, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 0, 4, float>) },
        { {false, 0, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 0, 5, float>) },
        { {false, 0, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 0, 6, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 1, 3, float>) },
        { {false, 1, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 1, 4, float>) },
        { {false, 1, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 1, 5, float>) },
        { {false, 1, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 1, 6, float>) },
        { {false, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 2, 2, float>) },
        { {false, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 2, 3, float>) },
        { {false, 2, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 2, 4, float>) },
        { {false, 2, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 2, 5, float>) },
        { {false, 2, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 2, 6, float>) },
        { {false, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 3, 2, float>) },
        { {false, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 3, 3, float>) },
        { {false, 3, 4, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 3, 4, float>) },
        { {false, 3, 5, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 3, 5, float>) },
        { {false, 3, 6, DT::F32}, reinterpret_cast<TFunc>(&DirectPairSummation<false, 3, 6, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in DirectPairSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 0, 4, float), (true, 0, 5, float), (true, 0, 6, float), (true, 1, 2, float), (true, 1, 3, float), (true, 1, 4, float), (true, 1, 5, float), (true, 1, 6, float), (true, 2, 2, float), (true, 2, 3, float), (true, 2, 4, float), (true, 2, 5, float), (true, 2, 6, float), (true, 3, 2, float), (true, 3, 3, float), (true, 3, 4, float), (true, 3, 5, float), (true, 3, 6, float), (false, 0, 2, float), (false, 0, 3, float), (false, 0, 4, float), (false, 0, 5, float), (false, 0, 6, float), (false, 1, 2, float), (false, 1, 3, float), (false, 1, 4, float), (false, 1, 5, float), (false, 1, 6, float), (false, 2, 2, float), (false, 2, 3, float), (false, 2, 4, float), (false, 2, 5, float), (false, 2, 6, float), (false, 3, 2, float), (false, 3, 3, float), (false, 3, 4, float), (false, 3, 5, float), (false, 3, 6, float)"
        );
    }
    const void* instance = it->second;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DirectPairSummationFFI, DirectPairSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Ret<ffi::AnyBuffer>() // loc_out
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdDirectPairSummation                    */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdDirectPairSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer gloc,
    ffi::AnyBuffer xm,
    ffi::AnyBuffer radial_kernel_params,
    ffi::Result<ffi::AnyBuffer> gxm,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int n = xm.dimensions()[0];
    int dim = xm.dimensions()[1] - 1;
    DT tvec = xm.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(xm.dimensions()[0], block_size));
    size_t smem = 2 * blockDim.x * (dim + 1) * (xm.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* gloc_arg = gloc.untyped_data();
    void* xm_arg = xm.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* gxm_arg = gxm->untyped_data();
    void* args[] = {
        &gloc_arg,
        &xm_arg,
        &radial_kernel_params_arg,
        &gxm_arg,
        &n
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 0, 3, float>) },
        { {true, 0, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 0, 4, float>) },
        { {true, 0, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 0, 5, float>) },
        { {true, 0, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 0, 6, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 1, 3, float>) },
        { {true, 1, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 1, 4, float>) },
        { {true, 1, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 1, 5, float>) },
        { {true, 1, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 1, 6, float>) },
        { {true, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 2, 2, float>) },
        { {true, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 2, 3, float>) },
        { {true, 2, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 2, 4, float>) },
        { {true, 2, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 2, 5, float>) },
        { {true, 2, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 2, 6, float>) },
        { {true, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 3, 2, float>) },
        { {true, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 3, 3, float>) },
        { {true, 3, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 3, 4, float>) },
        { {true, 3, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 3, 5, float>) },
        { {true, 3, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<true, 3, 6, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 0, 3, float>) },
        { {false, 0, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 0, 4, float>) },
        { {false, 0, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 0, 5, float>) },
        { {false, 0, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 0, 6, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 1, 3, float>) },
        { {false, 1, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 1, 4, float>) },
        { {false, 1, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 1, 5, float>) },
        { {false, 1, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 1, 6, float>) },
        { {false, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 2, 2, float>) },
        { {false, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 2, 3, float>) },
        { {false, 2, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 2, 4, float>) },
        { {false, 2, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 2, 5, float>) },
        { {false, 2, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 2, 6, float>) },
        { {false, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 3, 2, float>) },
        { {false, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 3, 3, float>) },
        { {false, 3, 4, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 3, 4, float>) },
        { {false, 3, 5, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 3, 5, float>) },
        { {false, 3, 6, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectPairSummation<false, 3, 6, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in BwdDirectPairSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 0, 4, float), (true, 0, 5, float), (true, 0, 6, float), (true, 1, 2, float), (true, 1, 3, float), (true, 1, 4, float), (true, 1, 5, float), (true, 1, 6, float), (true, 2, 2, float), (true, 2, 3, float), (true, 2, 4, float), (true, 2, 5, float), (true, 2, 6, float), (true, 3, 2, float), (true, 3, 3, float), (true, 3, 4, float), (true, 3, 5, float), (true, 3, 6, float), (false, 0, 2, float), (false, 0, 3, float), (false, 0, 4, float), (false, 0, 5, float), (false, 0, 6, float), (false, 1, 2, float), (false, 1, 3, float), (false, 1, 4, float), (false, 1, 5, float), (false, 1, 6, float), (false, 2, 2, float), (false, 2, 3, float), (false, 2, 4, float), (false, 2, 5, float), (false, 2, 6, float), (false, 3, 2, float), (false, 3, 3, float), (false, 3, 4, float), (false, 3, 5, float), (false, 3, 6, float)"
        );
    }
    const void* instance = it->second;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BwdDirectPairSummationFFI, BwdDirectPairSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // gloc
        .Arg<ffi::AnyBuffer>() // xm
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Ret<ffi::AnyBuffer>() // gxm
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: LeafLeafPairSummation                     */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error LeafLeafPairSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_recv,
    ffi::AnyBuffer spl_src,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_isrc,
    ffi::AnyBuffer posm_recv,
    ffi::AnyBuffer posm_src,
    ffi::AnyBuffer radial_kernel_params,
    ffi::Result<ffi::AnyBuffer> loc_recv,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int dim = posm_recv.dimensions()[1] - 1;
    DT tvec = posm_recv.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(spl_recv.element_count() - 1);
    size_t smem = blockDim.x * (dim + 1) * (posm_src.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_recv_arg = spl_recv.untyped_data();
    void* spl_src_arg = spl_src.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_isrc_arg = ilist_isrc.untyped_data();
    void* posm_recv_arg = posm_recv.untyped_data();
    void* posm_src_arg = posm_src.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* loc_recv_arg = loc_recv->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_recv_arg,
        &spl_src_arg,
        &spl_ilist_arg,
        &ilist_isrc_arg,
        &posm_recv_arg,
        &posm_src_arg,
        &radial_kernel_params_arg,
        &loc_recv_arg
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 1, 3, float>) },
        { {true, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 2, 2, float>) },
        { {true, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 2, 3, float>) },
        { {true, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 3, 2, float>) },
        { {true, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<true, 3, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 1, 3, float>) },
        { {false, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 2, 2, float>) },
        { {false, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 2, 3, float>) },
        { {false, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 3, 2, float>) },
        { {false, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairSummation<false, 3, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in LeafLeafPairSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (true, 2, 2, float), (true, 2, 3, float), (true, 3, 2, float), (true, 3, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float), (false, 2, 2, float), (false, 2, 3, float), (false, 3, 2, float), (false, 3, 3, float)"
        );
    }
    const void* instance = it->second;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LeafLeafPairSummationFFI, LeafLeafPairSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_recv
        .Arg<ffi::AnyBuffer>() // spl_src
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_isrc
        .Arg<ffi::AnyBuffer>() // posm_recv
        .Arg<ffi::AnyBuffer>() // posm_src
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Ret<ffi::AnyBuffer>() // loc_recv
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdLeafLeafPairSummation                  */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdLeafLeafPairSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_recv,
    ffi::AnyBuffer spl_src,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_isrc,
    ffi::AnyBuffer posm_recv,
    ffi::AnyBuffer posm_src,
    ffi::AnyBuffer radial_kernel_params,
    ffi::AnyBuffer gloc_recv,
    ffi::AnyBuffer gloc_src,
    ffi::Result<ffi::AnyBuffer> gposm_recv,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int dim = posm_recv.dimensions()[1] - 1;
    DT tvec = posm_recv.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(spl_recv.element_count() - 1);
    size_t smem = 2 * blockDim.x * (dim + 1) * (posm_src.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_recv_arg = spl_recv.untyped_data();
    void* spl_src_arg = spl_src.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_isrc_arg = ilist_isrc.untyped_data();
    void* posm_recv_arg = posm_recv.untyped_data();
    void* posm_src_arg = posm_src.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* gloc_recv_arg = gloc_recv.untyped_data();
    void* gloc_src_arg = gloc_src.untyped_data();
    void* gposm_recv_arg = gposm_recv->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_recv_arg,
        &spl_src_arg,
        &spl_ilist_arg,
        &ilist_isrc_arg,
        &posm_recv_arg,
        &posm_src_arg,
        &radial_kernel_params_arg,
        &gloc_recv_arg,
        &gloc_src_arg,
        &gposm_recv_arg
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 1, 3, float>) },
        { {true, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 2, 2, float>) },
        { {true, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 2, 3, float>) },
        { {true, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 3, 2, float>) },
        { {true, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<true, 3, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 1, 3, float>) },
        { {false, 2, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 2, 2, float>) },
        { {false, 2, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 2, 3, float>) },
        { {false, 3, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 3, 2, float>) },
        { {false, 3, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafPairSummation<false, 3, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in BwdLeafLeafPairSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (true, 2, 2, float), (true, 2, 3, float), (true, 3, 2, float), (true, 3, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float), (false, 2, 2, float), (false, 2, 3, float), (false, 3, 2, float), (false, 3, 3, float)"
        );
    }
    const void* instance = it->second;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BwdLeafLeafPairSummationFFI, BwdLeafLeafPairSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_recv
        .Arg<ffi::AnyBuffer>() // spl_src
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_isrc
        .Arg<ffi::AnyBuffer>() // posm_recv
        .Arg<ffi::AnyBuffer>() // posm_src
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Arg<ffi::AnyBuffer>() // gloc_recv
        .Arg<ffi::AnyBuffer>() // gloc_src
        .Ret<ffi::AnyBuffer>() // gposm_recv
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_pair_summation, m) {
    m.def("DirectPairSummation", []() { return EncapsulateFfiCall(&DirectPairSummationFFI); });
    m.def("BwdDirectPairSummation", []() { return EncapsulateFfiCall(&BwdDirectPairSummationFFI); });
    m.def("LeafLeafPairSummation", []() { return EncapsulateFfiCall(&LeafLeafPairSummationFFI); });
    m.def("BwdLeafLeafPairSummation", []() { return EncapsulateFfiCall(&BwdLeafLeafPairSummationFFI); });
}