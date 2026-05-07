library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity uart_tx is
    Generic (
        CLKS_PER_BIT : integer := 87 -- 10 MHz / 115200 Baud
    );
    Port (
        clk         : in  STD_LOGIC;
        rst         : in  STD_LOGIC;
        tx_start    : in  STD_LOGIC;
        tx_data     : in  STD_LOGIC_VECTOR(7 downto 0);
        tx_active   : out STD_LOGIC;
        tx_serial   : out STD_LOGIC;
        tx_done     : out STD_LOGIC
    );
end uart_tx;

architecture Behavioral of uart_tx is
    type t_SM_Main is (s_Idle, s_TX_Start_Bit, s_TX_Data_Bits, s_TX_Stop_Bit);
    signal r_SM_Main : t_SM_Main := s_Idle;
    
    signal r_Clk_Count : integer range 0 to CLKS_PER_BIT-1 := 0;
    signal r_Bit_Index : integer range 0 to 7 := 0;
    signal r_TX_Data   : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
    signal r_TX_Done   : STD_LOGIC := '0';
    
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                r_SM_Main <= s_Idle;
                tx_serial <= '1';
                tx_active <= '0';
                tx_done   <= '0';
            else
                case r_SM_Main is
                    when s_Idle =>
                        tx_active <= '0';
                        tx_serial <= '1';
                        r_TX_Done <= '0';
                        r_Clk_Count <= 0;
                        r_Bit_Index <= 0;
                        
                        if tx_start = '1' then
                            r_TX_Data <= tx_data;
                            r_SM_Main <= s_TX_Start_Bit;
                        end if;
                        
                    when s_TX_Start_Bit =>
                        tx_active <= '1';
                        tx_serial <= '0';
                        if r_Clk_Count < CLKS_PER_BIT-1 then
                            r_Clk_Count <= r_Clk_Count + 1;
                        else
                            r_Clk_Count <= 0;
                            r_SM_Main <= s_TX_Data_Bits;
                        end if;
                        
                    when s_TX_Data_Bits =>
                        tx_serial <= r_TX_Data(r_Bit_Index);
                        if r_Clk_Count < CLKS_PER_BIT-1 then
                            r_Clk_Count <= r_Clk_Count + 1;
                        else
                            r_Clk_Count <= 0;
                            if r_Bit_Index < 7 then
                                r_Bit_Index <= r_Bit_Index + 1;
                            else
                                r_Bit_Index <= 0;
                                r_SM_Main <= s_TX_Stop_Bit;
                            end if;
                        end if;
                        
                    when s_TX_Stop_Bit =>
                        tx_serial <= '1';
                        if r_Clk_Count < CLKS_PER_BIT-1 then
                            r_Clk_Count <= r_Clk_Count + 1;
                        else
                            r_TX_Done <= '1';
                            r_Clk_Count <= 0;
                            r_SM_Main <= s_Idle;
                        end if;
                        
                    when others =>
                        r_SM_Main <= s_Idle;
                end case;
            end if;
        end if;
    end process;
    
    tx_done <= r_TX_Done;
end Behavioral;