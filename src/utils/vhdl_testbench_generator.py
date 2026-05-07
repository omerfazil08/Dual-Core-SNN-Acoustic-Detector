# src/utils/vhdl_testbench_generator.py
import os, glob
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

# ==============================================================================
# RELATIVE PATHS (GitHub-Safe)
# ==============================================================================
TEST_BEBOP   = 'C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/data/splits/train_data_membo/test/bebop'
TEST_MEMBO   = 'C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/data/splits/train_data_membo/test/membo'
TEST_UNKNOWN = 'C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/data/splits/train_data_membo/test/unknown'

# Output directly into the Vivado simulation folder
OUTPUT_CSV   = 'C:/Users/omer-/Desktop/Dual-Core-SNN-Acoustic-Detector/hw/hdl/sim/drone_test_vectors_labeled_x.csv'

TARGET_SR = 16000
WINDOW_SIZE = 248  # ⚠️ UPDATED TO NEW 15.5ms MICRO-WINDOW

def process_file(path, label):
    try:
        sr, data = wavfile.read(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.empty((0, 2), dtype=int)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
        
    if sr != TARGET_SR:
        n_target = int(len(data) * TARGET_SR / sr)
        data = scipy_signal.resample(data, n_target)
        
    # Signed 8‑bit quantisation (AC pipeline)
    data = np.clip(np.round(data * 127.0), -127, 127).astype(np.int8)

    # Create windows
    n_windows = len(data) // WINDOW_SIZE
    data = data[:n_windows * WINDOW_SIZE]
    
    # Attach label
    labeled = np.zeros((len(data), 2), dtype=int)
    labeled[:, 0] = data
    labeled[:, 1] = label   # 1=Bebop, 2=Membo, 0=Noise
    
    return labeled

def main():
    print("🚀 Generating Dual-Core VHDL Testbench Vectors...")
    
    bebop_files = sorted(glob.glob(os.path.join(TEST_BEBOP, '*.wav')))
    membo_files = sorted(glob.glob(os.path.join(TEST_MEMBO, '*.wav')))
    noise_files = sorted(glob.glob(os.path.join(TEST_UNKNOWN, '*.wav')))

    all_blocks = []
    
    # 1. Process Bebop (Label 1)
    print(f"Processing {len(bebop_files)} Bebop files...")
    for f in bebop_files:
        all_blocks.append(process_file(f, 1))
        
    # 2. Process Membo (Label 2)
    print(f"Processing {len(membo_files)} Membo files...")
    for f in membo_files:
        all_blocks.append(process_file(f, 2))

    # 3. Process Noise (Label 0)
    print(f"Processing {len(noise_files)} Noise files...")
    for f in noise_files:
        all_blocks.append(process_file(f, 0))

    if not all_blocks:
        print("❌ Error: No audio files found. Check your relative paths!")
        return

    full = np.concatenate(all_blocks, axis=0)
    
    # Ensure the output directory exists before saving
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    np.savetxt(OUTPUT_CSV, full, fmt='%d,%d')
    
    print(f'✅ Written {len(full)} labelled samples to {OUTPUT_CSV}')

if __name__ == '__main__':
    main()