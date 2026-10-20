# ESP-DL helper for embedding aligned binary data
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)          # used inside utilities.cmake
include(${cmake_dir}/utilities.cmake)

idf_component_register(SRCS "app_main.cpp" INCLUDE_DIRS "." REQUIRES driver esp_timer)

# Model files written by ../quantize_export.py (INT8 convolution blocks + FP32 parameters); nothing else is embedded.
set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../model)
set(MODEL_FILES
    eeg_block_0.espdl eeg_block_1.espdl eog_block_0.espdl eog_block_1.espdl
    eeg_0_gn_weight.bin eeg_0_gn_bias.bin eeg_1_gn_weight.bin eeg_1_gn_bias.bin
    eog_0_gn_weight.bin eog_0_gn_bias.bin eog_1_gn_weight.bin eog_1_gn_bias.bin
    proj_linear_weight.bin proj_linear_bias.bin proj_ln_weight.bin proj_ln_bias.bin
    gru_weight_ih_l0.bin gru_weight_hh_l0.bin gru_bias_ih_l0.bin gru_bias_hh_l0.bin
    gru_norm_weight.bin gru_norm_bias.bin cls_ln_weight.bin cls_ln_bias.bin
    cls_linear_weight.bin cls_linear_bias.bin)
foreach(f ${MODEL_FILES})
    target_add_aligned_binary_data(${COMPONENT_LIB} ${MODEL_DIR}/${f} BINARY)
endforeach()
