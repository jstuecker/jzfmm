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

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SummarizeMultipoles                       */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error SummarizeMultipolesFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer mp_in,
    ffi::AnyBuffer xnode,
    ffi::AnyBuffer xchild,
    ffi::Result<ffi::AnyBuffer> mp_out,
    int p_in,
    bool kahan,
    int p
) {
    int nnodes = isplit.element_count() - 1;
    int dim = xnode.dimensions()[1];
    DT tvec = mp_in.element_type();
    dim3 blockDim(32);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, 32));
    size_t smem = 0;

    // Build a bundled argument list for cudaLaunchKernel
    void* isplit_arg = isplit.untyped_data();
    void* mp_in_arg = mp_in.untyped_data();
    void* xnode_arg = xnode.untyped_data();
    void* xchild_arg = xchild.untyped_data();
    void* mp_out_arg = mp_out->untyped_data();
    void* args[] = {
        &isplit_arg,
        &mp_in_arg,
        &xnode_arg,
        &xchild_arg,
        &mp_out_arg,
        &nnodes,
        &p_in,
        &kahan
    };


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {1, 2, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<1, 2, float>) },
        { {1, 2, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<1, 2, double>) },
        { {1, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<1, 3, float>) },
        { {1, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<1, 3, double>) },
        { {2, 2, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<2, 2, float>) },
        { {2, 2, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<2, 2, double>) },
        { {2, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<2, 3, float>) },
        { {2, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<2, 3, double>) },
        { {3, 2, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<3, 2, float>) },
        { {3, 2, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<3, 2, double>) },
        { {3, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<3, 3, float>) },
        { {3, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<3, 3, double>) },
        { {4, 2, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<4, 2, float>) },
        { {4, 2, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<4, 2, double>) },
        { {4, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<4, 3, float>) },
        { {4, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<4, 3, double>) },
        { {5, 2, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<5, 2, float>) },
        { {5, 2, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<5, 2, double>) },
        { {5, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<5, 3, float>) },
        { {5, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<5, 3, double>) },
        { {6, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<6, 3, float>) },
        { {6, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<6, 3, double>) },
        { {7, 3, DT::F32}, reinterpret_cast<TFunc>(&SummarizeMultipoles<7, 3, float>) },
        { {7, 3, DT::F64}, reinterpret_cast<TFunc>(&SummarizeMultipoles<7, 3, double>) }
    };

    const TTuple key = TTuple(p, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p, dim, tvec)"\
            " in SummarizeMultipolesFFIHost -- Only supporting:\n"\
            "(1, 2, float), (1, 2, double), (1, 3, float), (1, 3, double), (2, 2, float), (2, 2, double), (2, 3, float), (2, 3, double), (3, 2, float), (3, 2, double), (3, 3, float), (3, 3, double), (4, 2, float), (4, 2, double), (4, 3, float), (4, 3, double), (5, 2, float), (5, 2, double), (5, 3, float), (5, 3, double), (6, 3, float), (6, 3, double), (7, 3, float), (7, 3, double)"
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
    SummarizeMultipolesFFI, SummarizeMultipolesFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // mp_in
        .Arg<ffi::AnyBuffer>() // xnode
        .Arg<ffi::AnyBuffer>() // xchild
        .Ret<ffi::AnyBuffer>() // mp_out
        .Attr<int>("p_in")
        .Attr<bool>("kahan")
        .Attr<int>("p"),
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
    int dim = xnode.dimensions()[1];
    DT tvec = loc_node.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;

    // Initialize output buffers
    cudaMemsetAsync(loc_child->untyped_data(), 0, loc_child->size_bytes(), stream);

    // Build a bundled argument list for cudaLaunchKernel
    void* isplit_arg = isplit.untyped_data();
    void* loc_node_arg = loc_node.untyped_data();
    void* xnode_arg = xnode.untyped_data();
    void* xchild_arg = xchild.untyped_data();
    void* loc_child_arg = loc_child->untyped_data();
    void* args[] = {
        &isplit_arg,
        &loc_node_arg,
        &xnode_arg,
        &xchild_arg,
        &loc_child_arg,
        &nnodes,
        &pout
    };


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {1, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<1, 2, float>) },
        { {1, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<1, 2, double>) },
        { {1, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<1, 3, float>) },
        { {1, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<1, 3, double>) },
        { {2, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<2, 2, float>) },
        { {2, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<2, 2, double>) },
        { {2, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<2, 3, float>) },
        { {2, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<2, 3, double>) },
        { {3, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<3, 2, float>) },
        { {3, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<3, 2, double>) },
        { {3, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<3, 3, float>) },
        { {3, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<3, 3, double>) },
        { {4, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<4, 2, float>) },
        { {4, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<4, 2, double>) },
        { {4, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<4, 3, float>) },
        { {4, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<4, 3, double>) },
        { {5, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<5, 2, float>) },
        { {5, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<5, 2, double>) },
        { {5, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<5, 3, float>) },
        { {5, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<5, 3, double>) },
        { {6, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<6, 3, float>) },
        { {6, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<6, 3, double>) },
        { {7, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<7, 3, float>) },
        { {7, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal<7, 3, double>) }
    };

    const TTuple key = TTuple(p, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p, dim, tvec)"\
            " in TranslateLocalToLocalFFIHost -- Only supporting:\n"\
            "(1, 2, float), (1, 2, double), (1, 3, float), (1, 3, double), (2, 2, float), (2, 2, double), (2, 3, float), (2, 3, double), (3, 2, float), (3, 2, double), (3, 3, float), (3, 3, double), (4, 2, float), (4, 2, double), (4, 3, float), (4, 3, double), (5, 2, float), (5, 2, double), (5, 3, float), (5, 3, double), (6, 3, float), (6, 3, double), (7, 3, float), (7, 3, double)"
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
/*                             FFI call to CUDA kernel: TranslateLocalToLocal_XVJP                */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error TranslateLocalToLocal_XVJPFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer isplit,
    ffi::AnyBuffer loc_node,
    ffi::AnyBuffer xnode,
    ffi::AnyBuffer xchild,
    ffi::AnyBuffer g_loc_child,
    ffi::Result<ffi::AnyBuffer> g_xchild,
    int pout,
    int p,
    size_t block_size
) {
    int nnodes = isplit.element_count() - 1;
    int dim = xnode.dimensions()[1];
    DT tvec = loc_node.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(div_ceil(isplit.element_count() - 1, block_size));
    size_t smem = 0;

    // Build a bundled argument list for cudaLaunchKernel
    void* isplit_arg = isplit.untyped_data();
    void* loc_node_arg = loc_node.untyped_data();
    void* xnode_arg = xnode.untyped_data();
    void* xchild_arg = xchild.untyped_data();
    void* g_loc_child_arg = g_loc_child.untyped_data();
    void* g_xchild_arg = g_xchild->untyped_data();
    void* args[] = {
        &isplit_arg,
        &loc_node_arg,
        &xnode_arg,
        &xchild_arg,
        &g_loc_child_arg,
        &g_xchild_arg,
        &nnodes,
        &pout
    };


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<int, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {1, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<1, 2, float>) },
        { {1, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<1, 2, double>) },
        { {1, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<1, 3, float>) },
        { {1, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<1, 3, double>) },
        { {2, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<2, 2, float>) },
        { {2, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<2, 2, double>) },
        { {2, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<2, 3, float>) },
        { {2, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<2, 3, double>) },
        { {3, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<3, 2, float>) },
        { {3, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<3, 2, double>) },
        { {3, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<3, 3, float>) },
        { {3, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<3, 3, double>) },
        { {4, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<4, 2, float>) },
        { {4, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<4, 2, double>) },
        { {4, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<4, 3, float>) },
        { {4, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<4, 3, double>) },
        { {5, 2, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<5, 2, float>) },
        { {5, 2, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<5, 2, double>) },
        { {5, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<5, 3, float>) },
        { {5, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<5, 3, double>) },
        { {6, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<6, 3, float>) },
        { {6, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<6, 3, double>) },
        { {7, 3, DT::F32}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<7, 3, float>) },
        { {7, 3, DT::F64}, reinterpret_cast<TFunc>(&TranslateLocalToLocal_XVJP<7, 3, double>) }
    };

    const TTuple key = TTuple(p, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (p, dim, tvec)"\
            " in TranslateLocalToLocal_XVJPFFIHost -- Only supporting:\n"\
            "(1, 2, float), (1, 2, double), (1, 3, float), (1, 3, double), (2, 2, float), (2, 2, double), (2, 3, float), (2, 3, double), (3, 2, float), (3, 2, double), (3, 3, float), (3, 3, double), (4, 2, float), (4, 2, double), (4, 3, float), (4, 3, double), (5, 2, float), (5, 2, double), (5, 3, float), (5, 3, double), (6, 3, float), (6, 3, double), (7, 3, float), (7, 3, double)"
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
    TranslateLocalToLocal_XVJPFFI, TranslateLocalToLocal_XVJPFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // isplit
        .Arg<ffi::AnyBuffer>() // loc_node
        .Arg<ffi::AnyBuffer>() // xnode
        .Arg<ffi::AnyBuffer>() // xchild
        .Arg<ffi::AnyBuffer>() // g_loc_child
        .Ret<ffi::AnyBuffer>() // g_xchild
        .Attr<int>("pout")
        .Attr<int>("p")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_multipoles, m) {
    m.def("SummarizeMultipoles", []() { return EncapsulateFfiCall(&SummarizeMultipolesFFI); });
    m.def("TranslateLocalToLocal", []() { return EncapsulateFfiCall(&TranslateLocalToLocalFFI); });
    m.def("TranslateLocalToLocal_XVJP", []() { return EncapsulateFfiCall(&TranslateLocalToLocal_XVJPFFI); });
}