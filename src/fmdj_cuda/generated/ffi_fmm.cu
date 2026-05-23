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
#include "../fmm.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: CountInteractionsAndM2L                   */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error CountInteractionsAndM2LFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes_recv,
    ffi::AnyBuffer spl_nodes_src,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_isrc,
    ffi::AnyBuffer children_recv,
    ffi::AnyBuffer children_src,
    ffi::AnyBuffer mp_src,
    ffi::AnyBuffer radial_kernel_params,
    ffi::AnyBuffer opening_criterion_params,
    ffi::Result<ffi::AnyBuffer> loc_recv,
    ffi::Result<ffi::AnyBuffer> ilist_child_count_out,
    int radial_kernel_kind,
    int opening_criterion_kind,
    int p,
    int p_extra_m2l
) {
    int dim = children_recv.dimensions()[1] - 1;
    DT tvec = children_recv.element_type();
    dim3 blockDim(32);
    dim3 gridDim(spl_nodes_recv.element_count() - 1);
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(loc_recv->untyped_data(), 0, loc_recv->size_bytes(), stream);
    cudaMemsetAsync(ilist_child_count_out->untyped_data(), 0, ilist_child_count_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_recv_arg = spl_nodes_recv.untyped_data();
    void* spl_nodes_src_arg = spl_nodes_src.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_isrc_arg = ilist_isrc.untyped_data();
    void* children_recv_arg = children_recv.untyped_data();
    void* children_src_arg = children_src.untyped_data();
    void* mp_src_arg = mp_src.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* opening_criterion_params_arg = opening_criterion_params.untyped_data();
    void* loc_recv_arg = loc_recv->untyped_data();
    void* ilist_child_count_out_arg = ilist_child_count_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_recv_arg,
        &spl_nodes_src_arg,
        &spl_ilist_arg,
        &ilist_isrc_arg,
        &children_recv_arg,
        &children_src_arg,
        &mp_src_arg,
        &radial_kernel_params_arg,
        &opening_criterion_params_arg,
        &loc_recv_arg,
        &ilist_child_count_out_arg,
        &radial_kernel_kind
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {0, 1, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 0, 2, float>) },
        { {0, 1, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 0, 3, float>) },
        { {0, 1, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 1, 2, float>) },
        { {0, 1, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 1, 3, float>) },
        { {0, 2, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 0, 2, float>) },
        { {0, 2, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 0, 3, float>) },
        { {0, 2, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 1, 2, float>) },
        { {0, 2, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 1, 3, float>) },
        { {0, 3, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 0, 2, float>) },
        { {0, 3, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 0, 3, float>) },
        { {0, 3, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 1, 2, float>) },
        { {0, 3, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 1, 3, float>) },
        { {0, 4, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 0, 2, float>) },
        { {0, 4, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 0, 3, float>) },
        { {0, 4, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 1, 2, float>) },
        { {0, 4, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 1, 3, float>) },
        { {0, 5, 0, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 0, 2, float>) },
        { {0, 5, 0, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 0, 3, float>) },
        { {0, 5, 1, 2, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 1, 2, float>) },
        { {0, 5, 1, 3, DT::F32}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 1, 3, float>) }
    };

    const TTuple key = TTuple(opening_criterion_kind, p, p_extra_m2l, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (opening_criterion_kind, p, p_extra_m2l, dim, tvec)"\
            " in CountInteractionsAndM2LFFIHost -- Only supporting:\n"\
            "(0, 1, 0, 2, float), (0, 1, 0, 3, float), (0, 1, 1, 2, float), (0, 1, 1, 3, float), (0, 2, 0, 2, float), (0, 2, 0, 3, float), (0, 2, 1, 2, float), (0, 2, 1, 3, float), (0, 3, 0, 2, float), (0, 3, 0, 3, float), (0, 3, 1, 2, float), (0, 3, 1, 3, float), (0, 4, 0, 2, float), (0, 4, 0, 3, float), (0, 4, 1, 2, float), (0, 4, 1, 3, float), (0, 5, 0, 2, float), (0, 5, 0, 3, float), (0, 5, 1, 2, float), (0, 5, 1, 3, float)"
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
    CountInteractionsAndM2LFFI, CountInteractionsAndM2LFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes_recv
        .Arg<ffi::AnyBuffer>() // spl_nodes_src
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_isrc
        .Arg<ffi::AnyBuffer>() // children_recv
        .Arg<ffi::AnyBuffer>() // children_src
        .Arg<ffi::AnyBuffer>() // mp_src
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Arg<ffi::AnyBuffer>() // opening_criterion_params
        .Ret<ffi::AnyBuffer>() // loc_recv
        .Ret<ffi::AnyBuffer>() // ilist_child_count_out
        .Attr<int>("radial_kernel_kind")
        .Attr<int>("opening_criterion_kind")
        .Attr<int>("p")
        .Attr<int>("p_extra_m2l"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: InsertInteractions                        */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error InsertInteractionsFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes_recv,
    ffi::AnyBuffer spl_nodes_src,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_isrc,
    ffi::AnyBuffer children_recv,
    ffi::AnyBuffer children_src,
    ffi::AnyBuffer spl_ilist_child,
    ffi::AnyBuffer opening_criterion_params,
    ffi::Result<ffi::AnyBuffer> child_ilist_out,
    int opening_criterion_kind
) {
    int dim = children_recv.dimensions()[1] - 1;
    DT tvec = children_recv.element_type();
    dim3 blockDim(32);
    dim3 gridDim(spl_nodes_recv.element_count() - 1);
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_recv_arg = spl_nodes_recv.untyped_data();
    void* spl_nodes_src_arg = spl_nodes_src.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_isrc_arg = ilist_isrc.untyped_data();
    void* children_recv_arg = children_recv.untyped_data();
    void* children_src_arg = children_src.untyped_data();
    void* spl_ilist_child_arg = spl_ilist_child.untyped_data();
    void* opening_criterion_params_arg = opening_criterion_params.untyped_data();
    void* child_ilist_out_arg = child_ilist_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_recv_arg,
        &spl_nodes_src_arg,
        &spl_ilist_arg,
        &ilist_isrc_arg,
        &children_recv_arg,
        &children_src_arg,
        &spl_ilist_child_arg,
        &opening_criterion_params_arg,
        &child_ilist_out_arg
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {0, 2, DT::F32}, reinterpret_cast<TFunc>(&InsertInteractions<0, 2, float>) },
        { {0, 3, DT::F32}, reinterpret_cast<TFunc>(&InsertInteractions<0, 3, float>) }
    };

    const TTuple key = TTuple(opening_criterion_kind, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (opening_criterion_kind, dim, tvec)"\
            " in InsertInteractionsFFIHost -- Only supporting:\n"\
            "(0, 2, float), (0, 3, float)"
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
    InsertInteractionsFFI, InsertInteractionsFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_nodes_recv
        .Arg<ffi::AnyBuffer>() // spl_nodes_src
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_isrc
        .Arg<ffi::AnyBuffer>() // children_recv
        .Arg<ffi::AnyBuffer>() // children_src
        .Arg<ffi::AnyBuffer>() // spl_ilist_child
        .Arg<ffi::AnyBuffer>() // opening_criterion_params
        .Ret<ffi::AnyBuffer>() // child_ilist_out
        .Attr<int>("opening_criterion_kind"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fmm, m) {
    m.def("CountInteractionsAndM2L", []() { return EncapsulateFfiCall(&CountInteractionsAndM2LFFI); });
    m.def("InsertInteractions", []() { return EncapsulateFfiCall(&InsertInteractionsFFI); });
}