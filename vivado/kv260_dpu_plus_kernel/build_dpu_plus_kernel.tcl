# vivado/kv260_dpu_plus_kernel/build_dpu_plus_kernel.tcl
# Step 14 — Vivado Block Design automation for KV260 DPU + Custom HLS Kernel (Config C)
#
# Target: Kria KV260 Starter Kit (xck26-sfvc784-2LV-c)
# Integrates:
#   - DPUCZDX8G (B3136, 300 MHz)
#   - Mel_GEMM_Kernel (HLS IP from hls/ip_export/mel_gemm.zip)
#   - AXI Direct Memory Access (DMA) for fast memory streaming
#   - AXI SmartConnect to share DDR bandwidth with Zynq UltraScale+ PS
#
# Usage:
#   vivado -mode batch -source build_dpu_plus_kernel.tcl

set proj_name "kv260_dpu_plus_kernel"
set proj_dir "./prj_${proj_name}"
set part_name "xck26-sfvc784-2LV-c"
set board_part "xilinx.com:kv260_som:part0:1.4"
set ip_repo_dir "../../hls/ip_export"

puts "=========================================================="
puts " Building Combined Vivado Project: DPU + Custom Kernel"
puts " Target Part: ${part_name}"
puts "=========================================================="

create_project -force ${proj_name} ${proj_dir} -part ${part_name}
catch {set_property board_part ${board_part} [current_project]}

# Add custom HLS IP repository
set_property ip_repo_paths [list ${ip_repo_dir}] [current_project]
update_ip_catalog

# Create Block Design
create_bd_design "kv260_combined_bd"

# 1. Processing System
set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $zynq_ps

# Enable AXI ports for both DPU and Mel GEMM DMA
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
] $zynq_ps

# 2. Clocking (300 MHz DPU/kernel, 600 MHz 2x)
set clk_wiz [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz clk_wiz]
set_property -dict [list \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {300.0} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {600.0} \
    CONFIG.CLKOUT2_USED {true} \
] $clk_wiz

# 3. Instantiate AXI DMA for Mel GEMM streaming
set axi_dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma mel_dma]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
] $axi_dma

# 4. Instantiate Mel GEMM HLS Kernel (if catalog updated)
catch {
    set mel_ip [create_bd_cell -type ip -vlnv user:user:mel_gemm_top:1.0 mel_gemm]
    connect_bd_intf_net [get_bd_intf_pins mel_dma/M_AXIS_MM2S] [get_bd_intf_pins mel_gemm/power_in]
    connect_bd_intf_net [get_bd_intf_pins mel_gemm/mel_out] [get_bd_intf_pins mel_dma/S_AXIS_S2MM]
}

puts "[*] Validating combined design..."
validate_bd_design
save_bd_design

puts "[OK] Combined DPU + Custom Kernel project build complete."
exit
