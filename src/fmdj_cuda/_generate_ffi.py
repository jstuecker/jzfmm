from pathlib import Path
import os

import fmdj_utils.parse as parse
import fmdj_utils.generator as gen

HERE = Path(__file__).resolve().parent

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
kernels["GroupedForceAndPot"].smem_size_expression = "blockDim.x * sizeof(float4)"

kernels["BwdGroupedForceAndPot"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["BwdGroupedForceAndPot"].smem_size_expression = "2 * blockDim.x * sizeof(float4)"

kernels["ForceAndPotential"].grid_size_expression = "div_ceil(xm.element_count()/4, block_size)"
kernels["ForceAndPotential"].smem_size_expression = "blockDim.x * sizeof(float4)"
kernels["ForceAndPotential"].par["n"].expression = "xm.element_count()/4"

kernels["BwdForceAndPotential"].grid_size_expression = "div_ceil(xm.element_count()/4, block_size)"
kernels["BwdForceAndPotential"].smem_size_expression = "2 * blockDim.x * sizeof(float4)"
kernels["BwdForceAndPotential"].par["n"].expression = "xm.element_count()/4"


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

kernels["InsertInteractions"].grid_size_expression = "spl_nodes.element_count() - 1"
# kernels["InsertInteractions"].init_outputs_zero = True # this is actually expensive and not needed
kernels["InsertInteractions"].block_size_expression = 32

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

for kname in ("TranslateLocalToLocal", "CenterOfMass", "SummarizeMultipoles", "TranslateLocalToLocal_XVJP"):
    kernels[kname].grid_size_expression = "div_ceil(isplit.element_count() - 1, block_size)"
    kernels[kname].par["nnodes"].expression = "isplit.element_count() - 1"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_multipoles.cu"), 
    functions = kernels, 
    includes = default_includes + ["../multipoles.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                             tree.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "tree.cuh"),
    names=["PosZorderSort", "SummarizeLeaves", "FindNodeBoundaries", "GetNodeGeometry", "SearchSortedZ"],
    only_kernels=False
)

functions["PosZorderSort"].par["size"].expression = "pos_in.element_count()/3"
functions["PosZorderSort"].par["tmp_bytes"].expression = "tmp_buffer->size_bytes()"

functions["SummarizeLeaves"].par["n_leaves"].expression = "xnleaf.element_count()/4"
functions["SummarizeLeaves"].grid_size_expression = "div_ceil(n_leaves+1, block_size)"
functions["SummarizeLeaves"].smem_size_expression = "(block_size + 2*scan_size + 1) * (sizeof(PosN) + sizeof(int32_t))"

functions["FindNodeBoundaries"].par["size_nodes"].expression = "nodes_levels->element_count()"
functions["FindNodeBoundaries"].grid_size_expression = "div_ceil(size_nodes, block_size)"

functions["GetNodeGeometry"].par["size_nodes"].expression = "level->element_count()"
functions["GetNodeGeometry"].grid_size_expression = "div_ceil(size_nodes, block_size)"

functions["SearchSortedZ"].par["n_have"].expression = "posz_have.element_count()/3"
functions["SearchSortedZ"].par["n_query"].expression = "posz_query.element_count()/3"
functions["SearchSortedZ"].grid_size_expression = "div_ceil(n_query, block_size)"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_tree.cu"), 
    functions = functions, 
    includes = default_includes + ["../tree.cuh"]
)