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
#include "../forces.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: ForceAndPotential                         */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error ForceAndPotentialFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> loc_out,
    float epsilon,
    bool kahan,
    size_t block_size
) {
    int n = xm.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(xm.element_count()/4, block_size));
    size_t smem = blockDim.x * sizeof(float4);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* xm_arg = xm.untyped_data();
    void* loc_out_arg = loc_out->untyped_data();
    void* args[] = {
        &xm_arg,
        &loc_out_arg,
        &n,
        &epsilon
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true}, reinterpret_cast<TFunc>(&ForceAndPotential<true>) },
        { {false}, reinterpret_cast<TFunc>(&ForceAndPotential<false>) }
    };

    const TTuple key = TTuple(kahan);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in ForceAndPotentialFFIHost -- Only supporting:\n"\
            "(true), (false)"
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
    ForceAndPotentialFFI, ForceAndPotentialFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // loc_out
        .Attr<float>("epsilon")
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdForceAndPotential                      */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdForceAndPotentialFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer gloc,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> gxm,
    float epsilon,
    bool kahan,
    size_t block_size
) {
    int n = xm.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(xm.element_count()/4, block_size));
    size_t smem = 2 * blockDim.x * sizeof(float4);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* gloc_arg = gloc.untyped_data();
    void* xm_arg = xm.untyped_data();
    void* gxm_arg = gxm->untyped_data();
    void* args[] = {
        &gloc_arg,
        &xm_arg,
        &gxm_arg,
        &n,
        &epsilon
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true}, reinterpret_cast<TFunc>(&BwdForceAndPotential<true>) },
        { {false}, reinterpret_cast<TFunc>(&BwdForceAndPotential<false>) }
    };

    const TTuple key = TTuple(kahan);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in BwdForceAndPotentialFFIHost -- Only supporting:\n"\
            "(true), (false)"
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
    BwdForceAndPotentialFFI, BwdForceAndPotentialFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // gloc
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // gxm
        .Attr<float>("epsilon")
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: GroupedForceAndPot                        */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error GroupedForceAndPotFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer posm,
    ffi::Result<ffi::AnyBuffer> loc_out,
    float softening,
    bool kahan,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = blockDim.x * sizeof(float4);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* posm_arg = posm.untyped_data();
    void* loc_out_arg = loc_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &posm_arg,
        &loc_out_arg,
        &softening
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true}, reinterpret_cast<TFunc>(&GroupedForceAndPot<true>) },
        { {false}, reinterpret_cast<TFunc>(&GroupedForceAndPot<false>) }
    };

    const TTuple key = TTuple(kahan);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in GroupedForceAndPotFFIHost -- Only supporting:\n"\
            "(true), (false)"
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
    GroupedForceAndPotFFI, GroupedForceAndPotFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // posm
        .Ret<ffi::AnyBuffer>() // loc_out
        .Attr<float>("softening")
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdGroupedForceAndPot                     */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdGroupedForceAndPotFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer posm,
    ffi::AnyBuffer gloc,
    ffi::Result<ffi::AnyBuffer> gposm_out,
    float softening,
    bool kahan,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = 2 * blockDim.x * sizeof(float4);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* posm_arg = posm.untyped_data();
    void* gloc_arg = gloc.untyped_data();
    void* gposm_out_arg = gposm_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &posm_arg,
        &gloc_arg,
        &gposm_out_arg,
        &softening
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true}, reinterpret_cast<TFunc>(&BwdGroupedForceAndPot<true>) },
        { {false}, reinterpret_cast<TFunc>(&BwdGroupedForceAndPot<false>) }
    };

    const TTuple key = TTuple(kahan);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan)"\
            " in BwdGroupedForceAndPotFFIHost -- Only supporting:\n"\
            "(true), (false)"
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
    BwdGroupedForceAndPotFFI, BwdGroupedForceAndPotFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // posm
        .Arg<ffi::AnyBuffer>() // gloc
        .Ret<ffi::AnyBuffer>() // gposm_out
        .Attr<float>("softening")
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_forces, m) {
    m.def("ForceAndPotential", []() { return EncapsulateFfiCall(&ForceAndPotentialFFI); });
    m.def("BwdForceAndPotential", []() { return EncapsulateFfiCall(&BwdForceAndPotentialFFI); });
    m.def("GroupedForceAndPot", []() { return EncapsulateFfiCall(&GroupedForceAndPotFFI); });
    m.def("BwdGroupedForceAndPot", []() { return EncapsulateFfiCall(&BwdGroupedForceAndPotFFI); });
}