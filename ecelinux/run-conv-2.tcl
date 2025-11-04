#=============================================================================
# run_base.tcl 
#=============================================================================
# @brief: A Tcl script for synthesizing the baseline digit recongnition design.

# Project name
set hls_prj conv-2.prj

# Open/reset the project
open_project ${hls_prj} -reset

# Top function of the design is "dut"
set_top conv_layer_2

# Add design and testbench files
add_files bnn.cpp -cflags "-std=c++11"
add_files -tb bnn_test.cpp -cflags "-std=c++11"
add_files -tb data

open_solution "conv-2-resource-light-v4"
# Use Zynq device
set_part {xc7z020clg484-1}

# Target clock period is 10ns
create_clock -period 10

### You can insert your own directives here ###

############################################

# Simulate the C++ design
# csim_design -O
# Synthesize the design
csynth_design
# Co-simulate the design
#cosim_design
exit
