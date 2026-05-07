library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_weights_pkg is

    constant INPUT_SIZE  : integer := 1;
    constant HIDDEN_SIZE : integer := 64; -- UPDATED FOR DUAL-CORE

    type weight_array is array (0 to HIDDEN_SIZE-1) of signed(7 downto 0);
    type leak_array_type is array (0 to HIDDEN_SIZE-1) of integer range 0 to 7;

    -- =========================================================================
    -- CORE A: BEBOP RADAR (Hostile-Trained)
    -- =========================================================================
    constant BEBOP_T1 : signed(31 downto 0) := to_signed(1909, 32);
    constant BEBOP_T2 : signed(31 downto 0) := to_signed(27, 32);

    constant BEBOP_W1 : weight_array := (
        to_signed(-53, 8), to_signed(-2, 8), to_signed(-98, 8), to_signed(-68, 8), to_signed(-84, 8), to_signed(38, 8), to_signed(-60, 8), to_signed(-123, 8),
        to_signed(-46, 8), to_signed(48, 8), to_signed(-67, 8), to_signed(82, 8), to_signed(-44, 8), to_signed(-81, 8), to_signed(13, 8), to_signed(66, 8),
        to_signed(118, 8), to_signed(126, 8), to_signed(77, 8), to_signed(10, 8), to_signed(-76, 8), to_signed(-127, 8), to_signed(27, 8), to_signed(89, 8),
        to_signed(116, 8), to_signed(-126, 8), to_signed(31, 8), to_signed(-91, 8), to_signed(-118, 8), to_signed(-125, 8), to_signed(66, 8), to_signed(-112, 8),
        to_signed(66, 8), to_signed(-122, 8), to_signed(-88, 8), to_signed(-125, 8), to_signed(101, 8), to_signed(-43, 8), to_signed(87, 8), to_signed(50, 8),
        to_signed(-126, 8), to_signed(-5, 8), to_signed(-126, 8), to_signed(-51, 8), to_signed(26, 8), to_signed(100, 8), to_signed(94, 8), to_signed(24, 8),
        to_signed(14, 8), to_signed(-75, 8), to_signed(-81, 8), to_signed(-5, 8), to_signed(-121, 8), to_signed(-121, 8), to_signed(93, 8), to_signed(-111, 8),
        to_signed(37, 8), to_signed(73, 8), to_signed(127, 8), to_signed(-36, 8), to_signed(8, 8), to_signed(-16, 8), to_signed(56, 8), to_signed(-52, 8)
    );

    constant BEBOP_W2 : weight_array := (
        to_signed(-22, 8), to_signed(84, 8), to_signed(117, 8), to_signed(-51, 8), to_signed(110, 8), to_signed(28, 8), to_signed(62, 8), to_signed(-96, 8),
        to_signed(-83, 8), to_signed(-69, 8), to_signed(-117, 8), to_signed(11, 8), to_signed(-66, 8), to_signed(44, 8), to_signed(125, 8), to_signed(121, 8),
        to_signed(-23, 8), to_signed(127, 8), to_signed(5, 8), to_signed(-8, 8), to_signed(-52, 8), to_signed(34, 8), to_signed(-95, 8), to_signed(66, 8),
        to_signed(-99, 8), to_signed(48, 8), to_signed(14, 8), to_signed(-33, 8), to_signed(26, 8), to_signed(-55, 8), to_signed(51, 8), to_signed(125, 8),
        to_signed(-61, 8), to_signed(34, 8), to_signed(50, 8), to_signed(-84, 8), to_signed(-51, 8), to_signed(-110, 8), to_signed(-55, 8), to_signed(-83, 8),
        to_signed(70, 8), to_signed(4, 8), to_signed(57, 8), to_signed(54, 8), to_signed(46, 8), to_signed(-105, 8), to_signed(3, 8), to_signed(55, 8),
        to_signed(-88, 8), to_signed(31, 8), to_signed(53, 8), to_signed(-62, 8), to_signed(-120, 8), to_signed(-62, 8), to_signed(-60, 8), to_signed(-62, 8),
        to_signed(61, 8), to_signed(89, 8), to_signed(32, 8), to_signed(33, 8), to_signed(107, 8), to_signed(17, 8), to_signed(-101, 8), to_signed(-37, 8)
    );

    constant BEBOP_LEAKS : leak_array_type := (
        0, 0, 2, 2, 3, 0, 1, 0,
        1, 0, 2, 2, 1, 1, 2, 1,
        2, 3, 3, 2, 3, 3, 2, 3,
        2, 3, 2, 1, 3, 2, 2, 2,
        2, 3, 0, 0, 0, 2, 1, 1,
        0, 2, 0, 0, 2, 1, 2, 3,
        3, 3, 3, 2, 3, 3, 1, 2,
        3, 2, 1, 3, 2, 1, 2, 0
    );

    -- =========================================================================
    -- CORE B: MEMBO SNIPER (Standard-Trained)
    -- =========================================================================
    constant MEMBO_T1 : signed(31 downto 0) := to_signed(1464, 32);
    constant MEMBO_T2 : signed(31 downto 0) := to_signed(35, 32);

    constant MEMBO_W1 : weight_array := (
        to_signed(110, 8), to_signed(-26, 8), to_signed(-5, 8), to_signed(33, 8), to_signed(15, 8), to_signed(-58, 8), to_signed(-1, 8), to_signed(-24, 8),
        to_signed(70, 8), to_signed(63, 8), to_signed(-6, 8), to_signed(-64, 8), to_signed(-95, 8), to_signed(92, 8), to_signed(84, 8), to_signed(-84, 8),
        to_signed(36, 8), to_signed(14, 8), to_signed(56, 8), to_signed(-30, 8), to_signed(102, 8), to_signed(-112, 8), to_signed(44, 8), to_signed(42, 8),
        to_signed(-102, 8), to_signed(-45, 8), to_signed(-72, 8), to_signed(119, 8), to_signed(-110, 8), to_signed(-116, 8), to_signed(121, 8), to_signed(109, 8),
        to_signed(-83, 8), to_signed(106, 8), to_signed(77, 8), to_signed(-84, 8), to_signed(-9, 8), to_signed(-57, 8), to_signed(35, 8), to_signed(-73, 8),
        to_signed(-65, 8), to_signed(-63, 8), to_signed(75, 8), to_signed(-82, 8), to_signed(74, 8), to_signed(-34, 8), to_signed(87, 8), to_signed(-13, 8),
        to_signed(-38, 8), to_signed(-91, 8), to_signed(-122, 8), to_signed(-87, 8), to_signed(-119, 8), to_signed(31, 8), to_signed(28, 8), to_signed(83, 8),
        to_signed(-23, 8), to_signed(-5, 8), to_signed(102, 8), to_signed(78, 8), to_signed(-42, 8), to_signed(-116, 8), to_signed(20, 8), to_signed(-61, 8)
    );

    constant MEMBO_W2 : weight_array := (
        to_signed(47, 8), to_signed(-77, 8), to_signed(-28, 8), to_signed(-89, 8), to_signed(-39, 8), to_signed(19, 8), to_signed(9, 8), to_signed(-21, 8),
        to_signed(45, 8), to_signed(-95, 8), to_signed(-36, 8), to_signed(-9, 8), to_signed(96, 8), to_signed(58, 8), to_signed(-19, 8), to_signed(62, 8),
        to_signed(-24, 8), to_signed(-80, 8), to_signed(-73, 8), to_signed(7, 8), to_signed(72, 8), to_signed(81, 8), to_signed(117, 8), to_signed(35, 8),
        to_signed(10, 8), to_signed(6, 8), to_signed(-93, 8), to_signed(-2, 8), to_signed(-91, 8), to_signed(-22, 8), to_signed(-67, 8), to_signed(-13, 8),
        to_signed(106, 8), to_signed(-79, 8), to_signed(-57, 8), to_signed(-13, 8), to_signed(-36, 8), to_signed(55, 8), to_signed(63, 8), to_signed(23, 8),
        to_signed(-36, 8), to_signed(54, 8), to_signed(73, 8), to_signed(-35, 8), to_signed(-62, 8), to_signed(-29, 8), to_signed(-2, 8), to_signed(-57, 8),
        to_signed(-61, 8), to_signed(-32, 8), to_signed(-35, 8), to_signed(-15, 8), to_signed(-42, 8), to_signed(-52, 8), to_signed(120, 8), to_signed(-95, 8),
        to_signed(9, 8), to_signed(39, 8), to_signed(104, 8), to_signed(-93, 8), to_signed(0, 8), to_signed(14, 8), to_signed(42, 8), to_signed(-3, 8)
    );

    constant MEMBO_LEAKS : leak_array_type := (
        2, 3, 2, 2, 2, 0, 0, 2,
        2, 1, 1, 0, 3, 2, 0, 3,
        3, 1, 0, 3, 2, 2, 2, 0,
        1, 3, 1, 3, 0, 2, 0, 0,
        3, 3, 3, 0, 0, 2, 1, 1,
        3, 0, 1, 0, 3, 3, 0, 1,
        1, 3, 3, 2, 1, 3, 2, 0,
        1, 1, 3, 3, 3, 3, 3, 0
    );

end package snn_weights_pkg;
