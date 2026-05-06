import os, glob
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

# ADJUST THESE PATHS
TEST_BEBOP   = r'C:\Users\omer-\Desktop\drone_audio_detector\drone_analysis_data\test\bebop'
TEST_UNKNOWN = r'C:\Users\omer-\Desktop\drone_audio_detector\drone_analysis_data\test\unknown'
OUTPUT_CSV   = r'C:\Users\omer-\Desktop\drone_audio_detector\drone_analysis_data\vivado\sim\drone_test_vectors_labeled_x.csv'

TARGET_SR = 16000
WINDOW_SIZE = 1024

def process_file(path, label):
    sr, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    if sr != TARGET_SR:
        n_target = int(len(data) * TARGET_SR / sr)
        data = scipy_signal.resample(data, n_target)
    # Signed 8‑bit quantisation (AC pipeline), then absolute value
    data = np.clip(np.round(data * 127.0), -127, 127).astype(np.int8)
    #data = np.abs(data).astype(np.uint8)   # rectified 0‑127

    # Create windows
    n_windows = len(data) // WINDOW_SIZE
    data = data[:n_windows * WINDOW_SIZE]
    
    # Attach label (same label for every sample in the file)
    labeled = np.zeros((len(data), 2), dtype=int)
    labeled[:, 0] = data
    labeled[:, 1] = label   # 1 for drone, 0 for noise
    return labeled

def main():
    drone_files = sorted(glob.glob(os.path.join(TEST_BEBOP, '*.wav')))
    noise_files = sorted(glob.glob(os.path.join(TEST_UNKNOWN, '*.wav')))

    all_blocks = []
    for f in drone_files:
        all_blocks.append(process_file(f, 1))
    for f in noise_files:
        all_blocks.append(process_file(f, 0))

    full = np.concatenate(all_blocks, axis=0)
    np.savetxt(OUTPUT_CSV, full, fmt='%d,%d')
    print(f'Written {len(full)} labelled samples to {OUTPUT_CSV}')

if __name__ == '__main__':
    main()