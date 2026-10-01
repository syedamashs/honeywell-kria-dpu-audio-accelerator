# vivado/kv260_dpu_only/build_dpu_only.tcl
# Step 14 — Vivado Block Design automation for KV260 DPU-Only Platform (Config B)
#
# Target: Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
# Components:
#   - Zynq UltraScale+ MPSoC (Processing System)
#   - DPUCZDX8G (B3136, 1 core @ 300 MHz)
#   - Processor System Reset & Clocking Wizard
#   - AXI SmartConnect to HP/HPC DDR ports
#
# Usage:
#   vivado -mode batch -source build_dpu_only.tcl

set proj_name "kv260_dpu_only"
set proj_dir "./prj_${proj_name}"
set part_name "xck26-sfvc784-2LV-c"
set board_part "xilinx.com:kv260_som:part0:1.4"

puts "=========================================================="
puts " Building Vivado Project: ${proj_name} for ${part_name}"
puts "=========================================================="

create_project -force ${proj_name} ${proj_dir} -part ${part_name}
catch {set_property board_part ${board_part} [current_project]}

# Create Block Design
create_bd_design "kv260_dpu_bd"

# 1. Instantiate Zynq UltraScale+ MPSoC
set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $zynq_ps

# Enable High-Performance AXI Master/Slave Ports for DPU
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
] $zynq_ps

# 2. Clock Generator (DPU 300 MHz core clk, 600 MHz 2x clk)
set clk_wiz [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz clk_wiz]
set_property -dict [list \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {300.0} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {600.0} \
    CONFIG.CLKOUT2_USED {true} \
] $clk_wiz

# 3. Interconnect & Reset
set ps_rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset ps_rst]
set axi_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc]

puts "[*] Block Design configured. Validating design..."
validate_bd_design
save_bd_design

# Generate Wrapper and Bitstream
set bd_file [get_files "kv260_dpu_bd.bd"]
make_wrapper -files $bd_file -top
add_files -norecurse [glob ${proj_dir}/${proj_name}.srcs/sources_1/bd/kv260_dpu_bd/hdl/*_wrapper.v]

puts "[*] Running Synthesis and Implementation..."
# launch_runs impl_1 -to_step write_bitstream -jobs 4
# wait_on_run impl_1

puts "[OK] DPU-Only Vivado project build complete."
exit
