library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL; 

entity tb_snn_avionics_top is
end tb_snn_avionics_top;

architecture behavior of tb_snn_avionics_top is 

    component snn_avionics_top
    Port (
        clk              : in  STD_LOGIC;
        rst              : in  STD_LOGIC;
        adc_data_in      : in  STD_LOGIC_VECTOR(7 downto 0);
        data_valid       : in  STD_LOGIC;
        bebop_alarm_led  : out STD_LOGIC;
        membo_alarm_led  : out STD_LOGIC;
        uart_tx_pin      : out STD_LOGIC
    );
    end component;

    signal clk             : std_logic := '0';
    signal rst             : std_logic := '0';
    signal adc_data_in     : std_logic_vector(7 downto 0) := (others => '0');
    signal data_valid      : std_logic := '0';
    signal bebop_alarm_led : std_logic;
    signal membo_alarm_led : std_logic;
    signal uart_tx_pin     : std_logic; 

    -- 100 ns period = 10 MHz Clock
    constant clk_period : time := 100 ns;

begin

    uut: snn_avionics_top port map (
          clk             => clk,
          rst             => rst,
          adc_data_in     => adc_data_in,
          data_valid      => data_valid,
          bebop_alarm_led => bebop_alarm_led,
          membo_alarm_led => membo_alarm_led,
          uart_tx_pin     => uart_tx_pin
        );

    clk_process :process
    begin
        clk <= '0'; wait for clk_period/2;
        clk <= '1'; wait for clk_period/2;
    end process;

    stim_proc: process
        -- ?? UPDATE THIS PATH TO MATCH YOUR LOCAL MACHINE'S ABSOLUTE PATH
        file text_file : text open read_mode is "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/hw/hdl/sim/drone_test_vectors_labeled_x.csv";
        variable text_line : line;
        
        variable file_adc_val   : integer;
        variable comma_char     : character;
        variable file_label_val : integer;
        variable read_ok        : boolean; 
        
        variable sample_count : integer := 0; 
        
    begin       
        rst <= '1'; wait for 500 ns;   
        rst <= '0'; wait for clk_period * 10;

        report "--- STARTING DUAL-CORE HARDWARE SIMULATION ---";

        while not endfile(text_file) loop
            readline(text_file, text_line);
            
            read(text_line, file_adc_val, read_ok);
            if read_ok then
                read(text_line, comma_char, read_ok);
                read(text_line, file_label_val, read_ok);
                
                sample_count := sample_count + 1;
                
                adc_data_in <= std_logic_vector(to_signed(file_adc_val, 8));
                data_valid <= '1';
                wait for clk_period;
                
                data_valid <= '0';
                wait for clk_period * 5; 
                
                -- Print out alarms as they happen
                if bebop_alarm_led = '1' then
                    report "? BEBOP TARGET LOCKED at sample " & integer'image(sample_count);
                elsif membo_alarm_led = '1' then
                    report "? MEMBO TARGET LOCKED at sample " & integer'image(sample_count);
                end if;
                
            end if;
        end loop;
        
        wait for clk_period * 10;
        report "--- END OF SIMULATION ---";
        std.env.stop;
    end process;

end behavior;