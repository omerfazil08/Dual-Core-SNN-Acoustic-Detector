import numpy as np
import os

class Config:
    # ⚠️ Adjust these paths to point to your actual finalized weights
    BEBOP_W1 = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/bebop/final/best_bebop_weights_w1.npy"
    BEBOP_W2 = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/bebop/final/best_bebop_weights_w2.npy"
    BEBOP_GEN = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/bebop/final/best_bebop_genome_finetuned.npy"

    MEMBO_W1 = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/membo/final/best_mambo_weights_w1.npy"
    MEMBO_W2 = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/membo/final/best_mambo_weights_w2.npy"
    MEMBO_GEN = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/weights/membo/final/best_mambo_genome_finetuned.npy"

    # Where to save the generated VHDL file
    OUTPUT_FILE = "C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/hw/hdl/snn_weights_pkg.vhd"

def format_vhdl_array(name, arr, is_leak=False):
    lines = []
    array_type = "leak_array_type" if is_leak else "weight_array"
    lines.append(f"    constant {name} : {array_type} := (")
    
    # Chunk into rows of 8 for readability in VHDL
    for i in range(0, 64, 8):
        chunk = arr[i:i+8]
        if is_leak:
            row_str = ", ".join([str(int(x)) for x in chunk])
        else:
            row_str = ", ".join([f"to_signed({int(x)}, 8)" for x in chunk])
        
        if i + 8 < 64:
            row_str += ","
        lines.append(f"        {row_str}")
    lines.append("    );")
    return "\n".join(lines)

def main():
    print("🚀 Generating Dual-Core VHDL SNN Package...")

    # Load Bebop Data
    b_w1 = np.load(Config.BEBOP_W1)
    b_w2 = np.load(Config.BEBOP_W2)
    b_gen = np.load(Config.BEBOP_GEN)
    b_leaks, b_t1, b_t2 = b_gen[:64], b_gen[64], b_gen[65]

    # Load Membo Data
    m_w1 = np.load(Config.MEMBO_W1)
    m_w2 = np.load(Config.MEMBO_W2)
    m_gen = np.load(Config.MEMBO_GEN)
    m_leaks, m_t1, m_t2 = m_gen[:64], m_gen[64], m_gen[65]

    vhdl_template = f"""library IEEE;
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
    constant BEBOP_T1 : signed(31 downto 0) := to_signed({int(b_t1)}, 32);
    constant BEBOP_T2 : signed(31 downto 0) := to_signed({int(b_t2)}, 32);

{format_vhdl_array("BEBOP_W1", b_w1)}

{format_vhdl_array("BEBOP_W2", b_w2)}

{format_vhdl_array("BEBOP_LEAKS", b_leaks, is_leak=True)}

    -- =========================================================================
    -- CORE B: MEMBO SNIPER (Standard-Trained)
    -- =========================================================================
    constant MEMBO_T1 : signed(31 downto 0) := to_signed({int(m_t1)}, 32);
    constant MEMBO_T2 : signed(31 downto 0) := to_signed({int(m_t2)}, 32);

{format_vhdl_array("MEMBO_W1", m_w1)}

{format_vhdl_array("MEMBO_W2", m_w2)}

{format_vhdl_array("MEMBO_LEAKS", m_leaks, is_leak=True)}

end package snn_weights_pkg;
"""

    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    with open(Config.OUTPUT_FILE, 'w') as f:
        f.write(vhdl_template)
    
    print(f"✅ VHDL Package successfully written to: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()