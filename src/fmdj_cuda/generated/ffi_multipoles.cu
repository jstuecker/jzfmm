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
#include "../multipoles.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: MultipolesFromParticles                   */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error MultipolesFromParticlesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer part_posm,
    ffi::Result<ffi::AnyBuffer> mp_out,
    ffi::Result<ffi::AnyBuffer> xcom_out,
    bool around_com,
    int p,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(isplit.element_count() - 1);
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(mp_out->untyped_data(), 0, mp_out->size_bytes(), stream);
    cudaMemsetAsync(xcom_out->untyped_data(), 0, xcom_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    PosMass* part_posm_val = reinterpret_cast<PosMass*>(part_posm.untyped_data());
    float* mp_out_val = reinterpret_cast<float*>(mp_out->untyped_data());
    float3* xcom_out_val = reinterpret_cast<float3*>(xcom_out->untyped_data());

    void* args[] = {
        &isplit_val,
        &part_posm_val,
        &mp_out_val,
        &xcom_out_val,
        &around_com
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(MultipolesFromParticles<1>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{1}] = MultipolesFromParticles<1>;
    instance_map[{2}] = MultipolesFromParticles<2>;
    instance_map[{3}] = MultipolesFromParticles<3>;
    instance_map[{4}] = MultipolesFromParticles<4>;
    instance_map[{5}] = MultipolesFromParticles<5>;

    auto it = instance_map.find({p});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p)"\
            " in MultipolesFromParticlesFFIHost -- Only supporting:\n"\
            "(1), (2), (3), (4), (5)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MultipolesFromParticlesFFI, MultipolesFromParticlesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // part_posm
        .Ret<ffi::AnyBuffer>() // mp_out
        .Ret<ffi::AnyBuffer>() // xcom_out
        .Attr<bool>("around_com")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: CoarsenMultipoles                         */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error CoarsenMultipolesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer mp_values,
    ffi::AnyBuffer xcent,
    ffi::Result<ffi::AnyBuffer> mp_out,
    ffi::Result<ffi::AnyBuffer> xcent_out,
    bool around_com,
    int p,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(isplit.element_count() - 1);
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(mp_out->untyped_data(), 0, mp_out->size_bytes(), stream);
    cudaMemsetAsync(xcent_out->untyped_data(), 0, xcent_out->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    float* mp_values_val = reinterpret_cast<float*>(mp_values.untyped_data());
    float3* xcent_val = reinterpret_cast<float3*>(xcent.untyped_data());
    float* mp_out_val = reinterpret_cast<float*>(mp_out->untyped_data());
    float3* xcent_out_val = reinterpret_cast<float3*>(xcent_out->untyped_data());

    void* args[] = {
        &isplit_val,
        &mp_values_val,
        &xcent_val,
        &mp_out_val,
        &xcent_out_val,
        &around_com
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(CoarsenMultipoles<1>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{1}] = CoarsenMultipoles<1>;
    instance_map[{2}] = CoarsenMultipoles<2>;
    instance_map[{3}] = CoarsenMultipoles<3>;
    instance_map[{4}] = CoarsenMultipoles<4>;
    instance_map[{5}] = CoarsenMultipoles<5>;

    auto it = instance_map.find({p});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p)"\
            " in CoarsenMultipolesFFIHost -- Only supporting:\n"\
            "(1), (2), (3), (4), (5)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CoarsenMultipolesFFI, CoarsenMultipolesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // mp_values
        .Arg<ffi::AnyBuffer>() // xcent
        .Ret<ffi::AnyBuffer>() // mp_out
        .Ret<ffi::AnyBuffer>() // xcent_out
        .Attr<bool>("around_com")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: TranslateLocalToLocal                     */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error TranslateLocalToLocalFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer loc_node,
    ffi::AnyBuffer xnode,
    ffi::AnyBuffer xchild,
    ffi::Result<ffi::AnyBuffer> loc_child,
    int pout,
    int p,
    size_t block_size
) {
    int nnodes = isplit.element_count() - 1;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;
    
    // Initialize output buffers
    cudaMemsetAsync(loc_child->untyped_data(), 0, loc_child->size_bytes(), stream);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    float* loc_node_val = reinterpret_cast<float*>(loc_node.untyped_data());
    float3* xnode_val = reinterpret_cast<float3*>(xnode.untyped_data());
    float3* xchild_val = reinterpret_cast<float3*>(xchild.untyped_data());
    float* loc_child_val = reinterpret_cast<float*>(loc_child->untyped_data());

    void* args[] = {
        &isplit_val,
        &loc_node_val,
        &xnode_val,
        &xchild_val,
        &loc_child_val,
        &nnodes,
        &pout
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(TranslateLocalToLocal<1>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{1}] = TranslateLocalToLocal<1>;
    instance_map[{2}] = TranslateLocalToLocal<2>;
    instance_map[{3}] = TranslateLocalToLocal<3>;
    instance_map[{4}] = TranslateLocalToLocal<4>;
    instance_map[{5}] = TranslateLocalToLocal<5>;

    auto it = instance_map.find({p});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p)"\
            " in TranslateLocalToLocalFFIHost -- Only supporting:\n"\
            "(1), (2), (3), (4), (5)"
        );
    }

    TFunctionType* instance = it->second;
    
    cudaLaunchKernel((const void*)instance, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TranslateLocalToLocalFFI, TranslateLocalToLocalFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // loc_node
        .Arg<ffi::AnyBuffer>() // xnode
        .Arg<ffi::AnyBuffer>() // xchild
        .Ret<ffi::AnyBuffer>() // loc_child
        .Attr<int>("pout")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_multipoles, m) {
    m.def("MultipolesFromParticles", []() { return EncapsulateFfiCall(&MultipolesFromParticlesFFI); });
    m.def("CoarsenMultipoles", []() { return EncapsulateFfiCall(&CoarsenMultipolesFFI); });
    m.def("TranslateLocalToLocal", []() { return EncapsulateFfiCall(&TranslateLocalToLocalFFI); });
}