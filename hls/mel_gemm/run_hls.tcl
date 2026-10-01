# hls/mel_gemm/run_hls.tcl
# Step 13 — Vitis HLS automated flow for KV260 Mel Filterbank GEMM Kernel
#
# Usage:
#   vitis_hls -f run_hls.tcl

open_project -reset prj_mel_gemm
set_top mel_gemm_top

# Add kernel sources
add_files mel_gemm.cpp
add_files mel_weights_rom.cpp

# Add testbench
add_files -tb mel_gemm_tb.cpp

# Create solution targeting Kria KV260 (Zynq UltraScale+ xck26-sfvc784-2LV-c)
open_solution -reset "solution_kv260" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}

# Target clock: 300 MHz (3.33 ns period) matching KV260 DPU clock
create_clock -period 3.33 -name default

# Step 1: Run C Simulation
puts "=========================================================="
puts " Running C Simulation..."
puts "=========================================================="
csim_design

# Step 2: Run C Synthesis
puts "=========================================================="
puts " Running C Synthesis..."
puts "=========================================================="
csynth_design

# Step 3: Optional C/RTL Co-simulation
# cosim_design

# Step 4: Export RTL as Vivado IP Catalog IP
puts "=========================================================="
puts " Exporting Vivado IP Core..."
puts "=========================================================="
export_design -format ip_catalog -vendor "user" -version "1.0" -display_name "Mel_GEMM_Kernel" -output "../ip_export/mel_gemm.zip"

puts "=========================================================="
puts " Vitis HLS Flow Complete! IP exported to hls/ip_export/"
puts "=========================================================="
exit
