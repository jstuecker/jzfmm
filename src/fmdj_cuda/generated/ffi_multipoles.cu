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
/*                             FFI call to CUDA kernel: CenterOfMass                              */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error CenterOfMassFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer pos,
    ffi::AnyBuffer mass,
    ffi::Result<ffi::AnyBuffer> com_out,
    bool kahan,
    size_t block_size
) {
    int nnodes = isplit.element_count() - 1;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    float3* pos_val = reinterpret_cast<float3*>(pos.untyped_data());
    float* mass_val = reinterpret_cast<float*>(mass.untyped_data());
    float3* com_out_val = reinterpret_cast<float3*>(com_out->untyped_data());

    void* args[] = {
        &isplit_val,
        &pos_val,
        &mass_val,
        &com_out_val,
        &nnodes,
        &kahan
    };
    cudaLaunchKernel((const void*)CenterOfMass, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CenterOfMassFFI, CenterOfMassFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // pos
        .Arg<ffi::AnyBuffer>() // mass
        .Ret<ffi::AnyBuffer>() // com_out
        .Attr<bool>("kahan")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SummarizeMultipoles                       */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SummarizeMultipolesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer xnode,
    ffi::AnyBuffer xchild,
    ffi::AnyBuffer mp_in,
    ffi::Result<ffi::AnyBuffer> mp_out,
    int p_in,
    bool kahan,
    int p,
    size_t block_size
) {
    int nnodes = isplit.element_count() - 1;
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    int* isplit_val = reinterpret_cast<int*>(isplit.untyped_data());
    float3* xnode_val = reinterpret_cast<float3*>(xnode.untyped_data());
    float3* xchild_val = reinterpret_cast<float3*>(xchild.untyped_data());
    float* mp_in_val = reinterpret_cast<float*>(mp_in.untyped_data());
    float* mp_out_val = reinterpret_cast<float*>(mp_out->untyped_data());

    void* args[] = {
        &isplit_val,
        &xnode_val,
        &xchild_val,
        &mp_in_val,
        &mp_out_val,
        &nnodes,
        &p_in,
        &kahan
    };
    
    // We have template parameters, so we need to instantiate all valid templates
    // For this we select a function pointer through a map
    using TTuple = std::tuple<int>;
    using TFunctionType = decltype(SummarizeMultipoles<1>);

    std::map<TTuple, TFunctionType*> instance_map;
    instance_map[{1}] = SummarizeMultipoles<1>;
    instance_map[{2}] = SummarizeMultipoles<2>;
    instance_map[{3}] = SummarizeMultipoles<3>;
    instance_map[{4}] = SummarizeMultipoles<4>;
    instance_map[{5}] = SummarizeMultipoles<5>;

    auto it = instance_map.find({p});

    if(it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p)"\
            " in SummarizeMultipolesFFIHost -- Only supporting:\n"\
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
    SummarizeMultipolesFFI, SummarizeMultipolesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // xnode
        .Arg<ffi::AnyBuffer>() // xchild
        .Arg<ffi::AnyBuffer>() // mp_in
        .Ret<ffi::AnyBuffer>() // mp_out
        .Attr<int>("p_in")
        .Attr<bool>("kahan")
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
    m.def("CenterOfMass", []() { return EncapsulateFfiCall(&CenterOfMassFFI); });
    m.def("SummarizeMultipoles", []() { return EncapsulateFfiCall(&SummarizeMultipolesFFI); });
    m.def("TranslateLocalToLocal", []() { return EncapsulateFfiCall(&TranslateLocalToLocalFFI); });
}