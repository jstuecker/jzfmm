from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

dimensions = (2,3)
# float_types = ("float", "double")
float_types = ("float",) # by default don't compile double for now... doubles compilation time...
p_instance_values = (1, 2, 3, 4, 5)
p_l2l_instance_values = (1, 2, 3, 4, 5, 6)
p_extra_m2l_instance_values = (0, 1)
radial_kernel_instance_values = (0, 1)
opening_criterion_instance_values = (0,)
default_includes = ["../common/math.cuh"]

def add_dtype_template(func, buf_from, pos_types=float_types):
    func.template_par["tvec"].instances = pos_types
    func.template_par["tvec"].expression = f"{buf_from}.element_type()"

def dtype_size_expression(buf_from):
    return f"({buf_from}.element_type() == DT::F64 ? sizeof(double) : sizeof(float))"

# ------------------------------------------------------------------------------------------------ #
#                                            forces.cuh                                            #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "forces.cuh"), 
    only_kernels=True
)

kernels["LeafLeafSummation"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["LeafLeafSummation"].smem_size_expression = f"blockDim.x * (dim + 1) * {dtype_size_expression('posm')}"

kernels["BwdLeafLeafSummation"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["BwdLeafLeafSummation"].smem_size_expression = f"2 * blockDim.x * (dim + 1) * {dtype_size_expression('posm')}"

kernels["DirectSummation"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["DirectSummation"].smem_size_expression = f"blockDim.x * (dim + 1) * {dtype_size_expression('xm')}"
kernels["DirectSummation"].par["n"].expression = "xm.dimensions()[0]"

kernels["BwdDirectSummation"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["BwdDirectSummation"].smem_size_expression = f"2 * blockDim.x * (dim + 1) * {dtype_size_expression('xm')}"
kernels["BwdDirectSummation"].par["n"].expression = "xm.dimensions()[0]"

for kname in ("DirectSummation", "BwdDirectSummation"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "xm.dimensions()[1] - 1"
    kernels[kname].template_par["radial_kernel_kind"].instances = radial_kernel_instance_values
    add_dtype_template(kernels[kname], "xm")

for kname in ("LeafLeafSummation", "BwdLeafLeafSummation"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "posm.dimensions()[1] - 1"
    kernels[kname].template_par["radial_kernel_kind"].instances = radial_kernel_instance_values
    add_dtype_template(kernels[kname], "posm")

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_forces.cu"), 
    functions = kernels, 
    includes = default_includes + ["../forces.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                              fmm.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "fmm.cuh"), 
    only_kernels=True
)

kernels["CountInteractionsAndM2L"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["CountInteractionsAndM2L"].init_outputs_zero = True
kernels["CountInteractionsAndM2L"].block_size_expression = 32
kernels["CountInteractionsAndM2L"].template_par["p"].instances = p_instance_values
kernels["CountInteractionsAndM2L"].template_par["p_extra_m2l"].instances = p_extra_m2l_instance_values
kernels["CountInteractionsAndM2L"].template_par["opening_criterion_kind"].instances = opening_criterion_instance_values
kernels["CountInteractionsAndM2L"].template_par["dim"].instances = dimensions
kernels["CountInteractionsAndM2L"].template_par["dim"].expression = "children.dimensions()[1] - 1"
add_dtype_template(kernels["CountInteractionsAndM2L"], "children")

kernels["InsertInteractions"].grid_size_expression = "spl_nodes.element_count() - 1"
# kernels["InsertInteractions"].init_outputs_zero = True # this is actually expensive and not needed
kernels["InsertInteractions"].block_size_expression = 32
kernels["InsertInteractions"].template_par["opening_criterion_kind"].instances = opening_criterion_instance_values
kernels["InsertInteractions"].template_par["dim"].instances = dimensions
kernels["InsertInteractions"].template_par["dim"].expression = "children.dimensions()[1] - 1"
add_dtype_template(kernels["InsertInteractions"], "children")

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
for kname in ("TranslateLocalToLocal", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["p"].instances = p_l2l_instance_values

for kname in ("TranslateLocalToLocal",):
    kernels[kname].init_outputs_zero = True

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].grid_size_expression = "div_ceil(isplit.element_count() - 1, block_size)"
    kernels[kname].par["nnodes"].expression = "isplit.element_count() - 1"

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "xnode.dimensions()[1]"

add_dtype_template(kernels["SummarizeMultipoles"], "mp_in")
add_dtype_template(kernels["TranslateLocalToLocal"], "loc_node")
add_dtype_template(kernels["TranslateLocalToLocal_XVJP"], "loc_node")

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_multipoles.cu"), 
    functions = kernels, 
    includes = default_includes + ["../multipoles.cuh"]
)
