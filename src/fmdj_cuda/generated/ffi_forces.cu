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
/*                             FFI call to CUDA kernel: DirectSummation                           */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error DirectSummationFFIHost(
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
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<true, 1, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&DirectSummation<false, 1, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in DirectSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float)"
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
    DirectSummationFFI, DirectSummationFFIHost,
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
/*                             FFI call to CUDA kernel: BwdDirectSummation                        */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdDirectSummationFFIHost(
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
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<true, 1, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdDirectSummation<false, 1, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in BwdDirectSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float)"
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
    BwdDirectSummationFFI, BwdDirectSummationFFIHost,
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
/*                             FFI call to CUDA kernel: LeafLeafSummation                         */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error LeafLeafSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer posm,
    ffi::AnyBuffer radial_kernel_params,
    ffi::Result<ffi::AnyBuffer> loc_out,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int dim = posm.dimensions()[1] - 1;
    DT tvec = posm.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = blockDim.x * (dim + 1) * (posm.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* posm_arg = posm.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* loc_out_arg = loc_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &posm_arg,
        &radial_kernel_params_arg,
        &loc_out_arg
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<true, 1, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafSummation<false, 1, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in LeafLeafSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float)"
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
    LeafLeafSummationFFI, LeafLeafSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // posm
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Ret<ffi::AnyBuffer>() // loc_out
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: BwdLeafLeafSummation                      */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error BwdLeafLeafSummationFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer posm,
    ffi::AnyBuffer radial_kernel_params,
    ffi::AnyBuffer gloc,
    ffi::Result<ffi::AnyBuffer> gposm_out,
    bool kahan,
    int radial_kernel_kind,
    size_t block_size
) {
    int dim = posm.dimensions()[1] - 1;
    DT tvec = posm.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = 2 * blockDim.x * (dim + 1) * (posm.element_type() == DT::F64 ? sizeof(double) : sizeof(float));
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* posm_arg = posm.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* gloc_arg = gloc.untyped_data();
    void* gposm_out_arg = gposm_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &posm_arg,
        &radial_kernel_params_arg,
        &gloc_arg,
        &gposm_out_arg
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<true, 0, 2, float>) },
        { {true, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<true, 0, 3, float>) },
        { {true, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<true, 1, 2, float>) },
        { {true, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<true, 1, 3, float>) },
        { {false, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<false, 0, 2, float>) },
        { {false, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<false, 0, 3, float>) },
        { {false, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<false, 1, 2, float>) },
        { {false, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&BwdLeafLeafSummation<false, 1, 3, float>) }
    };

    const TTuple key = TTuple(kahan, radial_kernel_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (kahan, radial_kernel_kind, dim, tvec)"\
            " in BwdLeafLeafSummationFFIHost -- Only supporting:\n"\
            "(true, 0, 2, float), (true, 0, 3, float), (true, 1, 2, float), (true, 1, 3, float), (false, 0, 2, float), (false, 0, 3, float), (false, 1, 2, float), (false, 1, 3, float)"
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
    BwdLeafLeafSummationFFI, BwdLeafLeafSummationFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // posm
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Arg<ffi::AnyBuffer>() // gloc
        .Ret<ffi::AnyBuffer>() // gposm_out
        .Attr<bool>("kahan")
        .Attr<int>("radial_kernel_kind")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_forces, m) {
    m.def("DirectSummation", []() { return EncapsulateFfiCall(&DirectSummationFFI); });
    m.def("BwdDirectSummation", []() { return EncapsulateFfiCall(&BwdDirectSummationFFI); });
    m.def("LeafLeafSummation", []() { return EncapsulateFfiCall(&LeafLeafSummationFFI); });
    m.def("BwdLeafLeafSummation", []() { return EncapsulateFfiCall(&BwdLeafLeafSummationFFI); });
}