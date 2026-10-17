# Minimal package for GGML's Vulkan backend: the pinned SPIR-V headers (third_party/spirv-headers, fetched by
# scripts/bootstrap.ps1) are header-only, so no install step is needed.
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../third_party/spirv-headers/include")
endif()
set(SPIRV-Headers_FOUND TRUE)
