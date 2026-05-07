# ==============================================================================
# 1. CLOCK DEFINITION (10 MHz / 100ns)
# ==============================================================================
set_property PACKAGE_PIN W5 [get_ports clk]							
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 100.00 -waveform {0 50} [get_ports clk]

# ==============================================================================
# 2. RESET SIGNAL
# ==============================================================================
set_property PACKAGE_PIN U18 [get_ports rst]						
set_property IOSTANDARD LVCMOS33 [get_ports rst]

# ==============================================================================
# 3. ADC DATA INPUT (8-Bits: 0 to 255) & VALID SIGNAL
# ==============================================================================
set_property PACKAGE_PIN V17 [get_ports {adc_data_in[0]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[0]}]
set_property PACKAGE_PIN V16 [get_ports {adc_data_in[1]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[1]}]
set_property PACKAGE_PIN W16 [get_ports {adc_data_in[2]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[2]}]
set_property PACKAGE_PIN W17 [get_ports {adc_data_in[3]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[3]}]
set_property PACKAGE_PIN W15 [get_ports {adc_data_in[4]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[4]}]
set_property PACKAGE_PIN V15 [get_ports {adc_data_in[5]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[5]}]
set_property PACKAGE_PIN W14 [get_ports {adc_data_in[6]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[6]}]
set_property PACKAGE_PIN W13 [get_ports {adc_data_in[7]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[7]}]

set_property PACKAGE_PIN V2 [get_ports data_valid]					
set_property IOSTANDARD LVCMOS33 [get_ports data_valid]

# ==============================================================================
# 4. TARGET LOCKED ALARM LEDS (Bebop vs Membo)
# ==============================================================================
# Bebop mapped to main LED (J1)
set_property PACKAGE_PIN J1 [get_ports bebop_alarm_led]					
set_property IOSTANDARD LVCMOS33 [get_ports bebop_alarm_led]

# Membo mapped to secondary LED (L2)
set_property PACKAGE_PIN L2 [get_ports membo_alarm_led]					
set_property IOSTANDARD LVCMOS33 [get_ports membo_alarm_led]

# ==============================================================================
# 5. UART TELEMETRY TX PIN
# ==============================================================================
set_property PACKAGE_PIN A18 [get_ports uart_tx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx_pin]

# ==============================================================================
# 6. EXTERNAL I/O TIMING CONSTRAINTS
# ==============================================================================
set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports {adc_data_in[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports {adc_data_in[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports data_valid]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports data_valid]
set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports rst]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports rst]

set_output_delay -clock [get_clocks sys_clk_pin] -max 10.000 [get_ports bebop_alarm_led]
set_output_delay -clock [get_clocks sys_clk_pin] -min 1.000  [get_ports bebop_alarm_led]

set_output_delay -clock [get_clocks sys_clk_pin] -max 10.000 [get_ports membo_alarm_led]
set_output_delay -clock [get_clocks sys_clk_pin] -min 1.000  [get_ports membo_alarm_led]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

set_output_delay -clock [get_clocks sys_clk_pin] 0.000 [get_ports uart_tx_pin]