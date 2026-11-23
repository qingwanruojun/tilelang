file(SHA256 "${STAGE}/aicpu_kernels_device/libtilelang_aicpu_kernel.so" digest)
file(WRITE "${STAGE}/aicpu_kernels_device/bin_hash.cfg" "libtilelang_aicpu_kernel.so=${digest}\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar czf "${ARCHIVE}" aicpu_kernels_device
                WORKING_DIRECTORY "${STAGE}" RESULT_VARIABLE rc)
if(rc)
  message(FATAL_ERROR "Could not package the AICPU kernel: ${rc}")
endif()
