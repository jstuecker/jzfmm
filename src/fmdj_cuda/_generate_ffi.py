from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

dimensions = (2,3)
p_instance_values = (1, 2, 3, 4, 5)
default_includes = ["../common/math.cuh"]

# ------------------------------------------------------------------------------------------------ #
#                                            forces.cuh                                            #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "forces.cuh"), 
    only_kernels=True
)

kernels["GroupedForceAndPot"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["GroupedForceAndPot"].smem_size_expression = "blockDim.x * (dim + 1) * sizeof(float)"

kernels["BwdGroupedForceAndPot"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["BwdGroupedForceAndPot"].smem_size_expression = "2 * blockDim.x * (dim + 1) * sizeof(float)"

kernels["ForceAndPotential"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["ForceAndPotential"].smem_size_expression = "blockDim.x * (dim + 1) * sizeof(float)"
kernels["ForceAndPotential"].par["n"].expression = "xm.dimensions()[0]"

kernels["BwdForceAndPotential"].grid_size_expression = "div_ceil(xm.dimensions()[0], block_size)"
kernels["BwdForceAndPotential"].smem_size_expression = "2 * blockDim.x * (dim + 1) * sizeof(float)"
kernels["BwdForceAndPotential"].par["n"].expression = "xm.dimensions()[0]"

for kname in ("ForceAndPotential", "BwdForceAndPotential"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "xm.dimensions()[1] - 1"

for kname in ("GroupedForceAndPot", "BwdGroupedForceAndPot"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "posm.dimensions()[1] - 1"

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
kernels["CountInteractionsAndM2L"].template_par["dim"].instances = dimensions
kernels["CountInteractionsAndM2L"].template_par["dim"].expression = "children.dimensions()[1] - 1"

kernels["InsertInteractions"].grid_size_expression = "spl_nodes.element_count() - 1"
# kernels["InsertInteractions"].init_outputs_zero = True # this is actually expensive and not needed
kernels["InsertInteractions"].block_size_expression = 32
kernels["InsertInteractions"].template_par["dim"].instances = dimensions
kernels["InsertInteractions"].template_par["dim"].expression = "children.dimensions()[1] - 1"

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

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["p"].instances = p_instance_values

for kname in ("TranslateLocalToLocal",):
    kernels[kname].init_outputs_zero = True

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].grid_size_expression = "div_ceil(isplit.element_count() - 1, block_size)"
    kernels[kname].par["nnodes"].expression = "isplit.element_count() - 1"

for kname in ("TranslateLocalToLocal", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].template_par["dim"].instances = dimensions
    kernels[kname].template_par["dim"].expression = "xnode.dimensions()[1]"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_multipoles.cu"), 
    functions = kernels, 
    includes = default_includes + ["../multipoles.cuh"]
)
