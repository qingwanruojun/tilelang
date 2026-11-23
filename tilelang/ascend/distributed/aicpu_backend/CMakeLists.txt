cmake_minimum_required(VERSION 3.16)
project(tilelang_aicpu LANGUAGES CXX)
set(cann "${ASCEND_CANN_PACKAGE_PATH}")
add_library(tilelang_aicpu SHARED host.cpp resources.cpp)
target_compile_features(tilelang_aicpu PRIVATE cxx_std_17)
target_include_directories(tilelang_aicpu PRIVATE "${cann}/include" "${cann}/include/hccl" "${cann}/pkg_inc")
target_link_directories(tilelang_aicpu PRIVATE "${cann}/lib64")
target_link_libraries(tilelang_aicpu PRIVATE ascendcl hcomm hccl)

# Build the reusable device executor once for distribution in the platform OPP.
set(stage "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(soname libtilelang_aicpu_kernel.so)
set(kernel "${stage}/aicpu_kernels_device/${soname}")
set(archive "${CMAKE_CURRENT_BINARY_DIR}/aicpu_tilelang_transfer.tar.gz")
add_custom_command(OUTPUT "${kernel}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${stage}/aicpu_kernels_device"
  COMMAND "${cann}/tools/hcc/bin/aarch64-target-linux-gnu-g++"
          -std=c++17 -O2 -fPIC -fvisibility=hidden -shared -Wl,-z,defs
          "-I${cann}/include" "-I${cann}/include/hccl" "-L${cann}/devlib/device"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp" -lccl_kernel -o "${kernel}"
  DEPENDS kernel.cpp plan.h VERBATIM)
add_custom_command(OUTPUT "${archive}"
  COMMAND "${CMAKE_COMMAND}" "-DSTAGE=${stage}" "-DARCHIVE=${archive}" -P "${CMAKE_CURRENT_SOURCE_DIR}/package.cmake"
  DEPENDS "${kernel}" package.cmake VERBATIM)
add_custom_target(aicpu_package DEPENDS "${archive}")
