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
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer children,
    ffi::AnyBuffer mp_values,
    ffi::AnyBuffer radial_kernel_params,
    ffi::Result<ffi::AnyBuffer> loc_out,
    ffi::Result<ffi::AnyBuffer> ilist_child_count_out,
    float opening_angle,
    int radial_kernel_kind,
    int p
) {
    int dim = children.dimensions()[1] - 1;
    dim3 blockDim(32);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(loc_out->untyped_data(), 0, loc_out->size_bytes(), stream);
    cudaMemsetAsync(ilist_child_count_out->untyped_data(), 0, ilist_child_count_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* children_arg = children.untyped_data();
    void* mp_values_arg = mp_values.untyped_data();
    void* radial_kernel_params_arg = radial_kernel_params.untyped_data();
    void* loc_out_arg = loc_out->untyped_data();
    void* ilist_child_count_out_arg = ilist_child_count_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &children_arg,
        &mp_values_arg,
        &radial_kernel_params_arg,
        &loc_out_arg,
        &ilist_child_count_out_arg,
        &opening_angle
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, int>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {0, 1, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 2>) },
        { {0, 1, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 1, 3>) },
        { {0, 2, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 2>) },
        { {0, 2, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 2, 3>) },
        { {0, 3, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 2>) },
        { {0, 3, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 3, 3>) },
        { {0, 4, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 2>) },
        { {0, 4, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 4, 3>) },
        { {0, 5, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 2>) },
        { {0, 5, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<0, 5, 3>) },
        { {1, 1, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 1, 2>) },
        { {1, 1, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 1, 3>) },
        { {1, 2, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 2, 2>) },
        { {1, 2, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 2, 3>) },
        { {1, 3, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 3, 2>) },
        { {1, 3, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 3, 3>) },
        { {1, 4, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 4, 2>) },
        { {1, 4, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 4, 3>) },
        { {1, 5, 2}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 5, 2>) },
        { {1, 5, 3}, reinterpret_cast<TFunc>(&CountInteractionsAndM2L<1, 5, 3>) }
    };

    const TTuple key = TTuple(radial_kernel_kind, p, dim);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (radial_kernel_kind, p, dim)"\
            " in CountInteractionsAndM2LFFIHost -- Only supporting:\n"\
            "(0, 1, 2), (0, 1, 3), (0, 2, 2), (0, 2, 3), (0, 3, 2), (0, 3, 3), (0, 4, 2), (0, 4, 3), (0, 5, 2), (0, 5, 3), (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3), (1, 4, 2), (1, 4, 3), (1, 5, 2), (1, 5, 3)"
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
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // children
        .Arg<ffi::AnyBuffer>() // mp_values
        .Arg<ffi::AnyBuffer>() // radial_kernel_params
        .Ret<ffi::AnyBuffer>() // loc_out
        .Ret<ffi::AnyBuffer>() // ilist_child_count_out
        .Attr<float>("opening_angle")
        .Attr<int>("radial_kernel_kind")
        .Attr<int>("p"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: InsertInteractions                        */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error InsertInteractionsFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_nodes,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_nodes,
    ffi::AnyBuffer children,
    ffi::AnyBuffer spl_ilist_child,
    ffi::Result<ffi::AnyBuffer> child_ilist_out,
    float opening_angle
) {
    int dim = children.dimensions()[1] - 1;
    dim3 blockDim(32);
    dim3 gridDim(spl_nodes.element_count() - 1);
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_nodes_arg = spl_nodes.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_nodes_arg = ilist_nodes.untyped_data();
    void* children_arg = children.untyped_data();
    void* spl_ilist_child_arg = spl_ilist_child.untyped_data();
    void* child_ilist_out_arg = child_ilist_out->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_nodes_arg,
        &spl_ilist_arg,
        &ilist_nodes_arg,
        &children_arg,
        &spl_ilist_child_arg,
        &child_ilist_out_arg,
        &opening_angle
    };
    

    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {2}, reinterpret_cast<TFunc>(&InsertInteractions<2>) },
        { {3}, reinterpret_cast<TFunc>(&InsertInteractions<3>) }
    };

    const TTuple key = TTuple(dim);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (dim)"\
            " in InsertInteractionsFFIHost -- Only supporting:\n"\
            "(2), (3)"
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
        .Arg<ffi::AnyBuffer>() // spl_nodes
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_nodes
        .Arg<ffi::AnyBuffer>() // children
        .Arg<ffi::AnyBuffer>() // spl_ilist_child
        .Ret<ffi::AnyBuffer>() // child_ilist_out
        .Attr<float>("opening_angle"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_fmm, m) {
    m.def("CountInteractionsAndM2L", []() { return EncapsulateFfiCall(&CountInteractionsAndM2LFFI); });
    m.def("InsertInteractions", []() { return EncapsulateFfiCall(&InsertInteractionsFFI); });
}