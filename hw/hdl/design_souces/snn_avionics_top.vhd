library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity snn_avionics_top is
    Port (
        clk              : in  STD_LOGIC;
        rst              : in  STD_LOGIC;
        adc_data_in      : in  STD_LOGIC_VECTOR(7 downto 0);
        data_valid       : in  STD_LOGIC;
        
        -- Physical Debug Outputs
        bebop_alarm_led  : out STD_LOGIC; -- Maps to LED 1
        membo_alarm_led  : out STD_LOGIC; -- Maps to LED 2
        
        -- Avionics Telemetry
        uart_tx_pin      : out STD_LOGIC
    );
end snn_avionics_top;

architecture Behavioral of snn_avionics_top is

    -- 1. Component Declarations
    component neuromorphic_core is
        Generic (
            IS_BEBOP_CORE : boolean := true
        );
        Port (
            clk             : in  STD_LOGIC;
            rst             : in  STD_LOGIC;
            adc_data_in     : in  STD_LOGIC_VECTOR(7 downto 0);
            data_valid      : in  STD_LOGIC;
            target_locked   : out STD_LOGIC; 
            micro_alarm_out : out STD_LOGIC
        );
    end component;

    component uart_tx is
        Generic ( CLKS_PER_BIT : integer := 87 );
        Port (
            clk         : in  STD_LOGIC;
            rst         : in  STD_LOGIC;
            tx_start    : in  STD_LOGIC;
            tx_data     : in  STD_LOGIC_VECTOR(7 downto 0);
            tx_active   : out STD_LOGIC;
            tx_serial   : out STD_LOGIC;
            tx_done     : out STD_LOGIC
        );
    end component;

    -- 2. Internal Signals
    signal bebop_locked : STD_LOGIC;
    signal membo_locked : STD_LOGIC;
    
    -- Telemetry & Edge Detection
    signal prev_bebop_locked : STD_LOGIC := '0';
    signal prev_membo_locked : STD_LOGIC := '0';
    signal telemetry_start   : STD_LOGIC := '0';
    signal telemetry_data    : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
    signal uart_active       : STD_LOGIC;

begin

    -- =========================================================================
    -- CORE A: The Bebop Fortress (Hostile Trained)
    -- =========================================================================
    u_core_bebop : neuromorphic_core
    generic map ( IS_BEBOP_CORE => true )
    port map (
        clk             => clk,
        rst             => rst,
        adc_data_in     => adc_data_in,   -- Shared Audio Bus
        data_valid      => data_valid,    -- Shared Clock Enable
        target_locked   => bebop_locked,
        micro_alarm_out => open           -- Unused at top level
    );

    -- =========================================================================
    -- CORE B: The Membo Sniper (Clean Trained)
    -- =========================================================================
    u_core_membo : neuromorphic_core
    generic map ( IS_BEBOP_CORE => false )
    port map (
        clk             => clk,
        rst             => rst,
        adc_data_in     => adc_data_in,   -- Shared Audio Bus
        data_valid      => data_valid,    -- Shared Clock Enable
        target_locked   => membo_locked,
        micro_alarm_out => open
    );

    -- =========================================================================
    -- ASYMMETRIC PRIORITY MULTIPLEXER (Hardware Override)
    -- =========================================================================
    process(bebop_locked, membo_locked)
    begin
        -- Default to safe
        bebop_alarm_led <= '0';
        membo_alarm_led <= '0';
        
        if bebop_locked = '1' then
            -- Bebop absolutely overrides Membo
            bebop_alarm_led <= '1';
        elsif membo_locked = '1' then
            -- Membo only triggers if Bebop is silent
            membo_alarm_led <= '1';
        end if;
    end process;

    -- =========================================================================
    -- UART TELEMETRY TRANSMITTER
    -- =========================================================================
    u_telemetry_tx : uart_tx
    generic map ( CLKS_PER_BIT => 87 ) -- 10 MHz / 115200 Baud
    port map (
        clk       => clk,
        rst       => rst,
        tx_start  => telemetry_start,
        tx_data   => telemetry_data,
        tx_active => uart_active,
        tx_serial => uart_tx_pin,
        tx_done   => open
    );

    -- UART Edge Detection & Packet Formatting
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                prev_bebop_locked <= '0';
                prev_membo_locked <= '0';
                telemetry_start   <= '0';
                telemetry_data    <= (others => '0');
            else
                telemetry_start <= '0'; 
                
                if uart_active = '0' then
                    -- Priority 1: Bebop Rising Edge
                    if bebop_locked = '1' and prev_bebop_locked = '0' then
                        telemetry_data  <= x"42"; -- ASCII Hex for 'B' (Bebop)
                        telemetry_start <= '1';
                    
                    -- Priority 2: Membo Rising Edge (Only if Bebop isn't active)
                    elsif membo_locked = '1' and prev_membo_locked = '0' and bebop_locked = '0' then
                        telemetry_data  <= x"4D"; -- ASCII Hex for 'M' (Membo)
                        telemetry_start <= '1';
                    end if;
                end if;
                
                prev_bebop_locked <= bebop_locked;
                prev_membo_locked <= membo_locked;
            end if;
        end if;
    end process;

end Behavioral;