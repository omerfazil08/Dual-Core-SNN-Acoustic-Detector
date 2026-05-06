import os
import shutil
import glob
import numpy as np
import librosa
from scipy.io import wavfile
from collections import defaultdict

# --- CONFIGURATION ---
# 🔧 FIXED PATH to match the actual drone_analysis_data location
SOURCE_ROOT = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data/membo_phase'
DEST_ROOT = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data/membo_phase/drone_subsets_membo'
TARGET_SR = 16000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- FUNCTION TO ESTIMATE FUNDAMENTAL FREQUENCY ---
def get_fundamental_freq(filepath):
    try:
        sr, data = wavfile.read(filepath)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != TARGET_SR:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        else:
            data = data.astype(np.float32)

        data_norm = data / (np.max(np.abs(data)) + 1e-6)
        f0, voiced_flag, _ = librosa.pyin(data_norm, fmin=80, fmax=600, sr=TARGET_SR)
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            return np.median(f0)
    except Exception:
        pass
    return None

# --- FUNCTION TO CATEGORIZE NOISE FILES ---
def categorize_noise_file(filename):
    if 'white_noise' in filename:
        return 'white_noise'
    if 'pink_noise' in filename:
        return 'pink_noise'
    if 'silence' in filename:
        return 'silence'
    if 'running_tap' in filename or 'doing_the_dishes' in filename or 'exercise_bike' in filename:
        return 'domestic'
    return 'environmental'

# --- ANALYZE ALL TRAINING FILES ---
print("🔍 Analyzing training files...")
train_membo_path = os.path.join(SOURCE_ROOT, 'train', 'membo', '*.wav')
train_unknown_path = os.path.join(SOURCE_ROOT, 'train', 'unknown', '*.wav')

train_membo_files = sorted(glob.glob(train_membo_path))
train_noise_files = sorted(glob.glob(train_unknown_path))

print(f"Found {len(train_membo_files)} membo files.")
print(f"Found {len(train_noise_files)} Unknown files.")

if len(train_membo_files) == 0:
    print("❌ ERROR: No membo files found. Check SOURCE_ROOT path.")
    exit()
if len(train_noise_files) == 0:
    print("❌ ERROR: No noise files found. Check SOURCE_ROOT path.")
    exit()

membo_freqs = {}
for i, f in enumerate(train_membo_files):
    if i % 100 == 0:
        print(f"  Processing membo file {i+1}/{len(train_membo_files)}...")
    freq = get_fundamental_freq(f)
    if freq:
        membo_freqs[f] = freq

print(f"✅ Successfully analyzed {len(membo_freqs)} membo files.")

noise_categories = defaultdict(list)
for f in train_noise_files:
    cat = categorize_noise_file(os.path.basename(f))
    noise_categories[cat].append(f)
print(f"✅ Categorized {len(train_noise_files)} noise files into {len(noise_categories)} types.")

# --- SELECT REPRESENTATIVE membo FILES ---
avg_bpf = np.median(list(membo_freqs.values()))
print(f"\n📊 Average membo BPF: {avg_bpf:.1f} Hz")
sorted_membo = sorted(membo_freqs.items(), key=lambda item: abs(item[1] - avg_bpf))

# --- CREATE SUBSETS ---
subsets = {
    "1v1": 1,
    "5v5": 5,
    "10v10": 10,
    "50v50": 50,
    "100v100": 100
}

for name, count in subsets.items():
    print(f"\n📁 Creating subset: {name}")
    drone_dest = os.path.join(DEST_ROOT, name, 'drone')
    noise_dest = os.path.join(DEST_ROOT, name, 'noise')
    os.makedirs(drone_dest, exist_ok=True)
    os.makedirs(noise_dest, exist_ok=True)

    # Select membo files
    selected_membo = [f for f, _ in sorted_membo[:count]]
    for src in selected_membo:
        shutil.copy2(src, os.path.join(drone_dest, os.path.basename(src)))

    # Select diverse Noise files
    selected_noise = []
    if name == "1v1":
        if 'white_noise' in noise_categories and len(noise_categories['white_noise']) > 0:
            selected_noise = [noise_categories['white_noise'][0]]
        else:
            selected_noise = [train_noise_files[0]]  # fallback to first available
    else:
        cats = list(noise_categories.keys())
        for i in range(count):
            cat = cats[i % len(cats)]
            if noise_categories[cat]:
                f = noise_categories[cat].pop(0)
                selected_noise.append(f)
                noise_categories[cat].append(f)  # put back for round-robin

    for src in selected_noise:
        shutil.copy2(src, os.path.join(noise_dest, os.path.basename(src)))

    print(f"   → Copied {len(selected_membo)} drone and {len(selected_noise)} noise files.")

print(f"\n✅ All subsets created in: {DEST_ROOT}")