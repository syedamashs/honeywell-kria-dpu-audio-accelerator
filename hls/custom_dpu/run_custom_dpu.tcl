# ==============================================================================
# Vitis HLS 1-Click Synthesis Script: Custom DS-CNN Neural Network DPU IP Core
# Target: AMD Kria KV260 Starter Kit (xck26-sfvc784-2LV-c)
# Clock:  300.0 MHz (3.333 ns period)
# ==============================================================================

open_project -reset prj_custom_dpu
set_top custom_dpu_top

# Source files
add_files custom_dpu.cpp -cflags "-I."
add_files -tb custom_dpu_tb.cpp -cflags "-I."

open_solution -reset "solution1" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default

# Set hardware optimization directives
config_interface -m_axi_addr64
config_export -format ip_catalog -vendor "user" -version "1.0" -display_name "Custom_DSCNN_DPU_Core"

puts "=========================================================="
puts " Step 1: Running HLS C-Simulation (Golden Verification)"
puts "=========================================================="
csim_design

puts "=========================================================="
puts " Step 2: Running C-Synthesis (High-Level Synthesis @ 300MHz)"
puts "=========================================================="
csynth_design

puts "=========================================================="
puts " Step 3: Exporting Packaged Vivado IP Core"
puts "=========================================================="
export_design -format ip_catalog -description "Custom DS-CNN Depthwise Separable Neural Engine IP" -vendor "user" -version "1.0"

puts ">>> SUCCESS: Custom DPU IP Core exported to prj_custom_dpu/solution1/impl/ip <<<"
exit
