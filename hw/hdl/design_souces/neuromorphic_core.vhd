library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_weights_pkg.all; 

entity neuromorphic_core is
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
end neuromorphic_core;

architecture Behavioral of neuromorphic_core is

    constant MICRO_WINDOW_MAX : integer := 247;
    constant MACRO_WINDOW_MAX : integer := 63; -- 64 Micro-Windows = 0.99 seconds
    
    signal step_count         : integer range 0 to MICRO_WINDOW_MAX := 0;
    signal macro_frame_count  : integer range 0 to MACRO_WINDOW_MAX := 0;
    signal macro_spike_count  : integer range 0 to 64 := 0;

    type mem1_array is array (0 to 63) of signed(31 downto 0);
    signal mem1 : mem1_array := (others => (others => '0'));
    signal mem2 : signed(31 downto 0) := (others => '0');

    signal micro_alarm_flag  : STD_LOGIC := '0';
    signal tws_shift_reg     : STD_LOGIC_VECTOR(4 downto 0)  := (others => '0'); 

    -- =========================================================================
    -- THE PIPELINE STATE MACHINE
    -- =========================================================================
    type state_type is (S_IDLE, S_LAYER1, S_LAYER2, S_TWS);
    signal state : state_type := S_IDLE;
    
    signal latched_adc : signed(8 downto 0) := (others => '0');
    signal spk1_reg    : std_logic_vector(63 downto 0) := (others => '0');

begin

    process(clk)
        variable cur1, cur2      : signed(31 downto 0);
        variable mem1_leaked     : signed(31 downto 0);
        variable mem2_leaked     : signed(31 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state <= S_IDLE;
                step_count <= 0;
                macro_frame_count <= 0;
                macro_spike_count <= 0;
                
                mem1 <= (others => (others => '0'));
                mem2 <= (others => '0');
                micro_alarm_flag <= '0';
                tws_shift_reg   <= (others => '0');
                target_locked   <= '0';
                micro_alarm_out <= '0';
                latched_adc     <= (others => '0');
                spk1_reg        <= (others => '0');
                
            else
                case state is
                
                    -- CYCLE 1: WAIT FOR AUDIO
                    when S_IDLE =>
                        micro_alarm_out <= '0'; 
                        if data_valid = '1' then
                            latched_adc <= abs(resize(signed(adc_data_in), 9));
                            state <= S_LAYER1;
                        end if;

                    -- CYCLE 2: PROCESS 64 HIDDEN NEURONS
                    when S_LAYER1 =>
                        for i in 0 to 63 loop
                            if IS_BEBOP_CORE then
                                cur1 := resize(latched_adc * BEBOP_W1(i), 32);
                                mem1_leaked := shift_right(mem1(i), BEBOP_LEAKS(i));
                                if (mem1_leaked + cur1) > BEBOP_T1 then
                                    spk1_reg(i) <= '1';
                                    mem1(i) <= (others => '0'); 
                                else
                                    spk1_reg(i) <= '0';
                                    mem1(i) <= mem1_leaked + cur1; 
                                end if;
                            else
                                cur1 := resize(latched_adc * MEMBO_W1(i), 32);
                                mem1_leaked := shift_right(mem1(i), MEMBO_LEAKS(i));
                                if (mem1_leaked + cur1) > MEMBO_T1 then
                                    spk1_reg(i) <= '1';
                                    mem1(i) <= (others => '0'); 
                                else
                                    spk1_reg(i) <= '0';
                                    mem1(i) <= mem1_leaked + cur1; 
                                end if;
                            end if;
                        end loop;
                        state <= S_LAYER2;

                    -- CYCLE 3: PROCESS OUTPUT NEURON
                    when S_LAYER2 =>
                        cur2 := (others => '0');
                        for i in 0 to 63 loop
                            if spk1_reg(i) = '1' then
                                if IS_BEBOP_CORE then
                                    cur2 := cur2 + resize(BEBOP_W2(i), 32);
                                else
                                    cur2 := cur2 + resize(MEMBO_W2(i), 32);
                                end if;
                            end if;
                        end loop;

                        mem2_leaked := shift_right(mem2, 1);
                        
                        if IS_BEBOP_CORE then
                            if (mem2_leaked + cur2) > BEBOP_T2 then
                                mem2 <= (others => '0');
                                micro_alarm_flag <= '1'; 
                            else
                                mem2 <= mem2_leaked + cur2;
                            end if;
                        else
                            if (mem2_leaked + cur2) > MEMBO_T2 then
                                mem2 <= (others => '0');
                                micro_alarm_flag <= '1'; 
                            else
                                mem2 <= mem2_leaked + cur2;
                            end if;
                        end if;
                        state <= S_TWS;

                    -- CYCLE 4: TEMPORAL SLIDING WINDOWS (DISCRETE BLOCKS)
                    when S_TWS =>
                        if step_count = MICRO_WINDOW_MAX then
                            step_count <= 0;
                            mem1 <= (others => (others => '0'));
                            mem2 <= (others => '0');
                            
                            micro_alarm_out <= micro_alarm_flag;
                            
                            -- Accumulate spikes for this 1-second macro-window
                            if micro_alarm_flag = '1' then
                                macro_spike_count <= macro_spike_count + 1;
                            end if;
                            micro_alarm_flag <= '0';

                            -- End of 64-frame Macro Window (~1 second)
                            if macro_frame_count = MACRO_WINDOW_MAX then
                                macro_frame_count <= 0;
                                
                                -- Evaluate Coincidence ONLY once per second
                                if IS_BEBOP_CORE then
                                    if macro_spike_count >= 59 then
                                        tws_shift_reg <= tws_shift_reg(3 downto 0) & '1';
                                    else
                                        tws_shift_reg <= tws_shift_reg(3 downto 0) & '0';
                                    end if;
                                else
                                    if macro_spike_count >= 34 then
                                        tws_shift_reg <= tws_shift_reg(3 downto 0) & '1';
                                    else
                                        tws_shift_reg <= tws_shift_reg(3 downto 0) & '0';
                                    end if;
                                end if;
                                
                                macro_spike_count <= 0; -- Reset accumulator for next second
                            else
                                macro_frame_count <= macro_frame_count + 1;
                            end if;
                            
                            -- Absolute Lock Evaluation (Requires 5 consecutive seconds)
                            if tws_shift_reg = "11111" then
                                target_locked <= '1';
                            else
                                target_locked <= '0';
                            end if;

                        else
                            step_count <= step_count + 1;
                        end if;
                        
                        state <= S_IDLE;
                end case;
            end if;
        end if;
    end process;

end Behavioral;