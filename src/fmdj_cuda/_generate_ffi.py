from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

dimensions = (2,3)
direct_summation_dimensions = (2,3,4,5,6)
# float_types = ("float", "double")
float_types = ("float", "double") # by default don't compile double for now... doubles compilation time...
float_types_direct = ("float", "double")
p_instance_values = (1, 2, 3, 4, 5, 6, 7)
p_m2l_instance_values = (1, 2, 3, 4, 5, 6, 7)
p_l2l_instance_values = (1, 2, 3, 4, 5, 6, 7)
radial_kernel_instance_values = (0, 1, 2, 3)
opening_criterion_instance_values = (0,)
default_includes = ["../common/math.cuh"]

def add_dtype_template(func, buf_from, pos_types=float_types):
    func.template_par["tvec"].instances = pos_types
    func.template_par["tvec"].expression = f"{buf_from}.element_type()"

def dtype_size_expression(buf_from):
    return f"({buf_from}.element_type() == DT::F64 ? sizeof(double) : sizeof(float))"

def fmm_order_filter(*, p, dim, tvec, **_):
    """Limit expensive high orders to the configurations supported in practice."""
    return (p <= 5 or dim == 3) and (p <= 6 or tvec == "float")

# ------------------------------------------------------------------------------------------------ #
#                                        pair_summation.cuh                                       #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "pair_summation.cuh"), 
    only_kernels=True
)

kernels["LeafLeafPairSummation"].grid_size_expression = "spl_recv.element_count() - 1"
kernels["LeafLeafPairSummation"].smem_size_expression = f"blockDim.x * (dim + 1) * {dtype_size_expression('posm_src')}"

kernels["BwdLeafLeafPairSummation"].grid_size_expression = "spl_recv.element_count() - 1"
kernels["BwdLeafLeafPairSummation"].smem_size_expression = f"2 * blockDim.x * (dim + 1) * {dtype_size_expression('posm_src')}"

kernels["DirectPairSummation"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["DirectPairSummation"].smem_size_expression = f"blockDim.x * (dim + 1) * {dtype_size_expression('xm')}"
kernels["DirectPairSummation"].par["n"].expression = "xm.dimensions()[0]"

kernels["BwdDirectPairSummation"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["BwdDirectPairSummation"].smem_size_expression = f"2 * blockDim.x * (dim + 1) * {dtype_size_expression('xm')}"
kernels["BwdDirectPairSummation"].par["n"].expression = "xm.dimensions()[0]"

for kname in ("DirectPairSummation", "BwdDirectPairSummation"):
    kernels[kname].template_par["dim"].instances = direct_summation_dimensions
    kernels[kname].template_par["dim"].expression = "xm.dimensions()[1] - 1"
    kernels[kname].template_par["radial_kernel_kind"].instances = radial_kernel_instance_values
    add_dtype_template(kernels[kname], "xm", float_types_direct)

for kname in ("LeafLeafPairSummation", "BwdLeafLeafPairSummation"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "posm_recv.dimensions()[1] - 1"
    kernels[kname].template_par["radial_kernel_kind"].instances = radial_kernel_instance_values
    add_dtype_template(kernels[kname], "posm_recv", float_types_direct)

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_pair_summation.cu"), 
    functions = kernels, 
    includes = default_includes + ["../pair_summation.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                              fmm.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "fmm.cuh"), 
    only_kernels=True
)

kernels["CountInteractionsAndM2L"].grid_size_expression = "spl_nodes_recv.element_count() - 1"
kernels["CountInteractionsAndM2L"].block_size_expression = 32
kernels["CountInteractionsAndM2L"].par["ilist_child_count_out"].init_zero = True
kernels["CountInteractionsAndM2L"].template_par["p"].instances = p_m2l_instance_values
kernels["CountInteractionsAndM2L"].template_par["opening_criterion_kind"].instances = opening_criterion_instance_values
kernels["CountInteractionsAndM2L"].template_par["dim"].instances = dimensions
kernels["CountInteractionsAndM2L"].template_par["dim"].expression = "children_recv.dimensions()[1] - 1"
add_dtype_template(kernels["CountInteractionsAndM2L"], "children_recv")
kernels["CountInteractionsAndM2L"].template_filter = fmm_order_filter

kernels["InsertInteractions"].grid_size_expression = "spl_nodes_recv.element_count() - 1"
# kernels["InsertInteractions"].init_outputs_zero = True # this is actually expensive and not needed
kernels["InsertInteractions"].block_size_expression = 32
kernels["InsertInteractions"].template_par["opening_criterion_kind"].instances = opening_criterion_instance_values
kernels["InsertInteractions"].template_par["dim"].instances = dimensions
kernels["InsertInteractions"].template_par["dim"].expression = "children_recv.dimensions()[1] - 1"
add_dtype_template(kernels["InsertInteractions"], "children_recv")

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_fmm.cu"), 
    functions = kernels, 
    includes = default_includes + ["../fmm.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                          multipoles.cuh                                          #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "multipoles.cuh"), 
    only_kernels=True
)

kernels["SummarizeMultipoles"].template_par["p"].instances = p_instance_values
kernels["SummarizeMultipoles"].block_size_expression = 32
kernels["SummarizeMultipoles"].grid_size_expression = "div_ceil(isplit.element_count() - 1, 32)"
for kname in ("TranslateLocalToLocal", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["p"].instances = p_l2l_instance_values

for kname in ("TranslateLocalToLocal",):
    kernels[kname].init_outputs_zero = True

for kname in ("TranslateLocalToLocal", "TranslateLocalToLocal_XVJP"):
    kernels[kname].grid_size_expression = "div_ceil(isplit.element_count() - 1, block_size)"

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].par["nnodes"].expression = "isplit.element_count() - 1"

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "xnode.dimensions()[1]"
    kernels[kname].template_filter = fmm_order_filter

add_dtype_template(kernels["SummarizeMultipoles"], "mp_in")
add_dtype_template(kernels["TranslateLocalToLocal"], "loc_node")
add_dtype_template(kernels["TranslateLocalToLocal_XVJP"], "loc_node")

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_multipoles.cu"), 
    functions = kernels, 
    includes = default_includes + ["../multipoles.cuh"]
)
