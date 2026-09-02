cmake_minimum_required(VERSION 3.24)

# ================== Make sure we have a valid python interpreter =================================
# (We need it to define some paths)
if(DEFINED ENV{PYTHON_EXECUTABLE} AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "$ENV{PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable to use")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

if (NOT Python_Development.Module_FOUND)
  message(FATAL_ERROR
    "Python headers not found for ${Python_EXECUTABLE}.\n"
    "Recommended: Use a uv-managed Python:\n"
    "  uv python install 3.12 && uv python pin 3.12 && uv pip install -e . --no-build-isolation\n"
  )
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from nvidia import cu13; print(next(iter(cu13.__path__)))"
  OUTPUT_VARIABLE CUDA_PATH
  ERROR_VARIABLE CUDA_PATH_ERR
  RESULT_VARIABLE CUDA_PATH_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT CUDA_PATH_RESULT EQUAL 0)
  message(WARNING "Could not find python cuda directory - I will assume a system or conda/miniforge" 
          "-managed installation ") # ${CUDA_PATH_ERR}
else()
  # CUDA_ROOT defined via python we will try to set up CUDA paths from a python environment 
  message(STATUS "Using Python CUDA installation: ${CUDA_PATH}")

  set(_cuda_root "${CUDA_PATH}")
  set(_cuda_lib   "${_cuda_root}/lib")

  # Make sure the linker & runtime see the wheel libs during compiler detection
  if(DEFINED ENV{LIBRARY_PATH})
    set(ENV{LIBRARY_PATH} "${_cuda_lib}:$ENV{LIBRARY_PATH}")
  else()
    set(ENV{LIBRARY_PATH} "${_cuda_lib}")
  endif()

  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

  # Need to define the link directory explicitly here so that ninja finds it
  add_link_options(-L${_cuda_root}/lib)

  set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc")
endif()

# Somehow some conda environments like to put the env. variable NVCC_PREPEND_FLAGS and/or
# NVCC_PREPEND_FLAGS_BACKUP to "UNSET" which confuses nvcc. 
# Let's undefine these explicitly in that case:
foreach(var IN ITEMS NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP)
  if(DEFINED ENV{${var}} AND "$ENV{${var}}" STREQUAL "UNSET")
    message(STATUS "Clearing environment ${var} which was 'UNSET'")
    # CMake can't "unset" ENV vars, but setting to empty is equivalent for nvcc
    set(ENV{${var}} "")
  endif()
endforeach()

if(DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
  message(STATUS "User provided CUDAARCHS env: '$ENV{CUDAARCHS}'")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: '${CMAKE_CUDA_ARCHITECTURES}'")
else()
  message(STATUS "Using native CUDA architecture")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
  message(WARNING
    "Building jz-fmm for all supported CUDA architectures. This can take a very long time. "
    "Set CUDAARCHS=native or CUDAARCHS=<compute capability> to reduce compilation time."
  )
endif()

# Now let Cmake detect CUDA
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # To simplify finding paths in vscode

# Find FFI Paths
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_DIR
  ERROR_VARIABLE XLA_ERR
  RESULT_VARIABLE XLA_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get XLA include directory: ${XLA_ERR}")
endif()

message(STATUS "XLA include directory: ${XLA_DIR}")

# The p=7 compile-time M2L contraction exceeds NVCC's default template depth.
# The following ptxas flag also reports register and local-memory usage.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ftemplate-depth=512 -Xptxas=-v")

include_directories(SYSTEM ${XLA_DIR})


# ----------------------------------------------------------------------------------------------- #
# Auto-add all modules from src/jzfmm_cuda/generated
# ----------------------------------------------------------------------------------------------- #

option(JZFMM_GENERATE "Generate FFI source files at configure time" OFF)
if(DEFINED ENV{JZFMM_GENERATE} AND
   ("$ENV{JZFMM_GENERATE}" STREQUAL "1" OR
    "$ENV{JZFMM_GENERATE}" STREQUAL "ON" OR
    "$ENV{JZFMM_GENERATE}" STREQUAL "TRUE"))
  set(JZFMM_GENERATE ON)
endif()

# Optionally run the generator so files exist at configure time (opt-in)
if(JZFMM_GENERATE)
  message(STATUS "JZFMM_GENERATE=ON: Running CUDA FFI generator script")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/jzfmm_cuda/_generate_ffi.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
else()
  message(STATUS "JZFMM_GENERATE=OFF: Skipping CUDA FFI generation (enable with -DJZFMM_GENERATE=ON or JZFMM_GENERATE=1)")
endif()

# Pick up all generated .cu files; re-scan on re-configure
file(GLOB GEN_FFI_CU CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/jzfmm_cuda/generated/*.cu"
)

foreach(GEN_SRC IN LISTS GEN_FFI_CU)
  message(STATUS "Compiling generated FFI module from source: ${GEN_SRC}")
  get_filename_component(MOD_NAME "${GEN_SRC}" NAME_WE)  # e.g. ffi_fmm.cu -> ffi_fmm

  nanobind_add_module(${MOD_NAME} "${GEN_SRC}")
  target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_compile_features(${MOD_NAME} PRIVATE cxx_std_17)

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}_cuda)
endforeach()
