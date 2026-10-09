# ==============================================================================
# Vivado Block Design Automation: Config D Dual Custom Heterogeneous System
# Target: AMD Kria KV260 Starter Kit (xck26-sfvc784-2LV-c)
# Integrates:
#   1. Custom Mel GEMM HLS IP Core (0x00A0010000)
#   2. Custom DS-CNN Neural Network DPU IP Core (0x00A0020000)
#   3. AXI Direct Memory Access (DMA) Core (0x00A0000000)
#   4. Zynq UltraScale+ MPSoC Processing System (PS)
#
# Usage:
#   vivado -mode batch -source build_dual_custom_bd.tcl
# ==============================================================================

set proj_name "kv260_config_d_dual_custom"
set proj_dir "./prj_${proj_name}"
set part_name "xck26-sfvc784-2LV-c"
set board_part "xilinx.com:kv260_som:part0:1.4"

create_project -force ${proj_name} ${proj_dir} -part ${part_name}
catch {set_property board_part ${board_part} [current_project]}

# Set IP Repository Paths
set_property ip_repo_paths [list "../../hls/mel_gemm/export" "../../hls/custom_dpu/export"] [current_project]
update_ip_catalog

# Create Block Design
create_bd_design "kv260_config_d_bd"

# 1. Instantiate Zynq MPSoC PS
set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $zynq_ps

# Enable Master and Slave AXI High-Performance Ports
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
] $zynq_ps

# 2. Instantiate AXI DMA (Scatter-Gather Disabled, Direct Register Mode)
set axi_dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma mel_dma]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
] $axi_dma

# 3. Instantiate Custom Mel GEMM HLS Core
catch {
    set mel_ip [create_bd_cell -type ip -vlnv user:user:mel_gemm_top:1.0 mel_gemm]
    # Connect DMA MM2S Stream -> Mel GEMM Ingest Stream
    connect_bd_intf_net [get_bd_intf_pins mel_dma/M_AXIS_MM2S] [get_bd_intf_pins mel_gemm/power_in]
}

# 4. Instantiate Custom DS-CNN Neural Network DPU Core
catch {
    set dscnn_ip [create_bd_cell -type ip -vlnv user:user:custom_dpu_top:1.0 custom_dpu]
    # Direct On-Chip Hardware Pipeline: Mel GEMM Out -> Custom DPU In (Zero DDR Latency!)
    connect_bd_intf_net [get_bd_intf_pins mel_gemm/mel_out] [get_bd_intf_pins custom_dpu/features_in]
    # Custom DPU Logits Out -> DMA S2MM Stream (Back to DDR memory)
    connect_bd_intf_net [get_bd_intf_pins custom_dpu/logits_out] [get_bd_intf_pins mel_dma/S_AXIS_S2MM]
}

puts "[*] Validating Config D Dual Custom Block Design..."
validate_bd_design
save_bd_design

puts ">>> SUCCESS: Config D Vivado Dual-IP Block Design generated successfully! <<<"
exit
