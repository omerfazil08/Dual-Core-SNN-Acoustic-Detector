import os
import shutil
import glob
import random
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SOURCE_MEMBO = 'C:/Users/omer-/Desktop/drone_audio_detector/data/DroneAudioDataset-master/Multiclass_Drone_Audio/membo_1'
SOURCE_UNKNOWN = 'C:/Users/omer-/Desktop/drone_audio_detector/data/DroneAudioDataset-master/Binary_Drone_Audio/unknown'
DEST_ROOT = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data/membo_phase'

TRAIN_RATIO = 0.80
RANDOM_SEED = 42   # For reproducibility


# ==============================================================================
# 1. GATHER ALL FILES
# ==============================================================================
membo_files = sorted(glob.glob(os.path.join(SOURCE_MEMBO, '*.wav')))
unknown_files = sorted(glob.glob(os.path.join(SOURCE_UNKNOWN, '*.wav')))

print(f"Found {len(membo_files)} membo files.")
print(f"Found {len(unknown_files)} Unknown files.")

# Shuffle deterministically
random.seed(RANDOM_SEED)
random.shuffle(membo_files)
random.shuffle(unknown_files)

# Balance classes: use same number of unknown as membo
unknown_files = unknown_files[:len(membo_files)]
print(f"Balanced to {len(membo_files)} files per class.")

# ==============================================================================
# 2. TRAIN/TEST SPLIT
# ==============================================================================
split_idx_membo = int(len(membo_files) * TRAIN_RATIO)
split_idx_unknown = int(len(unknown_files) * TRAIN_RATIO)

train_membo = membo_files[:split_idx_membo]
test_membo = membo_files[split_idx_membo:]

train_unknown = unknown_files[:split_idx_unknown]
test_unknown = unknown_files[split_idx_unknown:]

print(f"\nTrain: {len(train_membo)} membo + {len(train_unknown)} Unknown = {len(train_membo)+len(train_unknown)} files")
print(f"Test:  {len(test_membo)} membo + {len(test_unknown)} Unknown = {len(test_membo)+len(test_unknown)} files")

# ==============================================================================
# 3. CREATE DESTINATION FOLDERS AND COPY FILES
# ==============================================================================
for split, subdir in [('train', 'train'), ('test', 'test')]:
    for cls, folder in [('membo', 'membo'), ('unknown', 'unknown')]:
        os.makedirs(os.path.join(DEST_ROOT, subdir, folder), exist_ok=True)

manifest_lines = []
for split_name, membo_list, unknown_list in [
    ('train', train_membo, train_unknown),
    ('test', test_membo, test_unknown)
]:
    for cls_name, file_list in [('membo', membo_list), ('unknown', unknown_list)]:
        for src in file_list:
            fname = os.path.basename(src)
            dst = os.path.join(DEST_ROOT, split_name, cls_name, fname)
            shutil.copy2(src, dst)
            manifest_lines.append(f"{split_name},{cls_name},{fname}")

# ==============================================================================
# 4. SAVE MANIFEST
# ==============================================================================
manifest_path = os.path.join(DEST_ROOT, 'file_manifest.txt')
with open(manifest_path, 'w') as f:
    f.write("split,class,filename\n")
    f.write("\n".join(manifest_lines))

print(f"\n✅ Files copied to: {DEST_ROOT}")
print(f"✅ Manifest saved to: {manifest_path}")