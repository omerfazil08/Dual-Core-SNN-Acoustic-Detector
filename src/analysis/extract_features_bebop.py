import os
import json
import glob
import numpy as np
import librosa
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DRONE_ANALYSIS_ROOT = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data'
SUBSETS_ROOT = os.path.join(DRONE_ANALYSIS_ROOT, 'drone_subsets')
MASTER_ANALYSIS_DIR = os.path.join(DRONE_ANALYSIS_ROOT, 'master_analysis')
TARGET_SR = 16000
WINDOW_SIZE = 1024   # 64 ms
HOP_SIZE = 512
NUM_MFCC = 13
HIGH_PASS_CUTOFF = 60  # Hz
NUM_HARMONICS = 4

os.makedirs(MASTER_ANALYSIS_DIR, exist_ok=True)

# ==============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==============================================================================
def extract_features(audio, sr):
    """Extract 21‑dim feature vector from raw audio (already normalized)."""
    audio = audio.astype(np.float32)
    
    # Time‑domain
    zcr = float(librosa.feature.zero_crossing_rate(audio, frame_length=len(audio), center=False)[0, 0])
    rms = float(librosa.feature.rms(y=audio, frame_length=len(audio), center=False)[0, 0])
    
    # Spectral features
    stft = np.abs(librosa.stft(audio, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=WINDOW_SIZE)
    
    centroid = float(librosa.feature.spectral_centroid(S=stft, freq=freqs)[0, 0])
    rolloff = float(librosa.feature.spectral_rolloff(S=stft, freq=freqs, roll_percent=0.85)[0, 0])
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE)
    mfccs = mfccs[:, 0].astype(np.float32)
    
    return {
        'zcr': zcr,
        'rms': rms,
        'centroid': centroid,
        'rolloff': rolloff,
        'mfccs': mfccs,
        'stft': stft[:, 0],
        'freqs': freqs
    }

def compute_average_spectrum(wav_files, max_files=200):
    """Compute average magnitude spectrum across many files."""
    avg_spec = None
    count = 0
    for f in wav_files[:max_files]:
        try:
            sr, data = wavfile.read(f)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != TARGET_SR:
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            data = data / (np.max(np.abs(data)) + 1e-6)
            stft = np.abs(librosa.stft(data, n_fft=WINDOW_SIZE, hop_length=WINDOW_SIZE))
            mean_spec = np.mean(stft, axis=1)
            if avg_spec is None:
                avg_spec = mean_spec
            else:
                avg_spec += mean_spec
            count += 1
        except Exception:
            continue
    if count > 0:
        avg_spec /= count
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=WINDOW_SIZE)
    return freqs, avg_spec

def find_harmonic_bands(freqs, avg_spec, num_harmonics=4):
    """Find peaks in the average spectrum and define bands."""
    # High‑pass filter
    mask = freqs >= HIGH_PASS_CUTOFF
    freqs_filt = freqs[mask]
    spec_filt = avg_spec[mask]
    
    if spec_filt is None or len(spec_filt) == 0:
        return _fallback_bands()
    
    # Relaxed peak finding
    height_thresh = np.max(spec_filt) * 0.05  # Lower threshold
    peaks, _ = scipy.signal.find_peaks(spec_filt, height=height_thresh, distance=5)
    
    if len(peaks) < num_harmonics:
        # If not enough peaks, use the strongest available
        peaks = np.argsort(spec_filt)[-num_harmonics*2:]
        peaks = peaks[spec_filt[peaks] > height_thresh]
    
    if len(peaks) == 0:
        return _fallback_bands()
    
    peak_freqs = freqs_filt[peaks]
    peak_amps = spec_filt[peaks]
    sorted_idx = np.argsort(peak_amps)[::-1]
    
    bands = []
    for i in range(min(num_harmonics, len(sorted_idx))):
        f_center = peak_freqs[sorted_idx[i]]
        half_width = max(15, f_center * 0.1)
        bands.append((float(f_center - half_width), float(f_center + half_width)))
    
    bands.sort(key=lambda x: x[0])
    
    # Ensure we have exactly num_harmonics bands
    while len(bands) < num_harmonics:
        # Add fallback bands
        fallback = _fallback_bands()
        for fb in fallback:
            if fb not in bands:
                bands.append(fb)
                break
    return bands[:num_harmonics]

def _fallback_bands():
    return [(80.0, 120.0), (160.0, 240.0), (280.0, 360.0), (400.0, 480.0)]

def compute_relative_band_energies(stft, freqs, bands):
    """Compute relative energy in each band."""
    energies = []
    total = np.sum(stft ** 2) + 1e-6
    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        band_energy = np.sum(stft[mask] ** 2)
        energies.append(float(band_energy / total))
    return np.array(energies, dtype=np.float32)

# ==============================================================================
# 2. PROCESS A SINGLE SUBSET
# ==============================================================================
def analyze_subset(subset_path, subset_name):
    print(f"\n{'='*60}")
    print(f"📊 Analyzing subset: {subset_name}")
    print(f"{'='*60}")
    
# Handle both naming conventions
    if os.path.exists(os.path.join(subset_path, 'bebop')):
        drone_folder = os.path.join(subset_path, 'bebop')
        noise_folder = os.path.join(subset_path, 'unknown')
    else:
        drone_folder = os.path.join(subset_path, 'drone')
        noise_folder = os.path.join(subset_path, 'noise')
    
    drone_files = glob.glob(os.path.join(drone_folder, '*.wav'))
    noise_files = glob.glob(os.path.join(noise_folder, '*.wav'))
    
    if not drone_files or not noise_files:
        print(f"⚠️ Missing files in {subset_name}, skipping.")
        return None
    
    # Step 1: Compute average Bebop spectrum and harmonic bands
    print("🔍 Computing average Bebop spectrum...")
    freqs, avg_spec = compute_average_spectrum(drone_files, max_files=min(50, len(drone_files)))
    bands = find_harmonic_bands(freqs, avg_spec, NUM_HARMONICS)
    print(f"   → Harmonic bands: {bands}")
    
    # Step 2: Extract features for all windows
    def process_file_list(file_list, label):
        features = []
        for f in file_list:
            try:
                sr, data = wavfile.read(f)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr != TARGET_SR:
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
                data = data / (np.max(np.abs(data)) + 1e-6)
                
                n_windows = len(data) // WINDOW_SIZE
                for i in range(n_windows):
                    win = data[i*WINDOW_SIZE : (i+1)*WINDOW_SIZE]
                    feats = extract_features(win, TARGET_SR)
                    rel_bands = compute_relative_band_energies(feats['stft'], feats['freqs'], bands)
                    feat_vec = np.concatenate([
                        [feats['zcr'], feats['rms'], feats['centroid'], feats['rolloff']],
                        rel_bands,
                        feats['mfccs']
                    ]).astype(np.float32)
                    features.append(feat_vec)
            except Exception:
                continue
        return np.array(features, dtype=np.float32) if features else np.array([], dtype=np.float32)
    
    print("📈 Extracting features from drone files...")
    drone_feats = process_file_list(drone_files, 1)
    print(f"   → {len(drone_feats)} drone windows")
    print("📈 Extracting features from noise files...")
    noise_feats = process_file_list(noise_files, 0)
    print(f"   → {len(noise_feats)} noise windows")
    
    if len(drone_feats) == 0 or len(noise_feats) == 0:
        print("⚠️ No features extracted.")
        return None
    
    # Step 3: Save features and statistics
    analysis_dir = os.path.join(subset_path, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    np.save(os.path.join(analysis_dir, 'features_drone.npy'), drone_feats)
    np.save(os.path.join(analysis_dir, 'features_noise.npy'), noise_feats)
    
    feature_names = ['ZCR', 'RMS', 'Centroid', 'Rolloff'] + [f'Band_{i+1}' for i in range(NUM_HARMONICS)] + [f'MFCC_{i+1}' for i in range(NUM_MFCC)]
    
    # Convert to Python native types for JSON
    drone_mean = drone_feats.mean(axis=0).tolist()
    drone_std = drone_feats.std(axis=0).tolist()
    noise_mean = noise_feats.mean(axis=0).tolist()
    noise_std = noise_feats.std(axis=0).tolist()
    
    band_indices = [4 + i for i in range(NUM_HARMONICS)]
    band_thresholds = [float(np.percentile(noise_feats[:, idx], 90)) for idx in band_indices]
    
    stats = {
        'subset_name': subset_name,
        'num_drone_windows': int(len(drone_feats)),
        'num_noise_windows': int(len(noise_feats)),
        'harmonic_bands': bands,
        'drone_mean': drone_mean,
        'drone_std': drone_std,
        'noise_mean': noise_mean,
        'noise_std': noise_std,
        'band_thresholds_90pct': band_thresholds
    }
    
    with open(os.path.join(analysis_dir, 'stats_summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Step 4: Generate visualizations
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(feature_names[:8]):
        if i >= drone_feats.shape[1]:
            break
        plt.subplot(2, 4, i+1)
        plt.hist(drone_feats[:, i], bins=30, alpha=0.5, label='Drone', density=True)
        plt.hist(noise_feats[:, i], bins=30, alpha=0.5, label='Noise', density=True)
        plt.title(name)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'feature_distributions.png'), dpi=150)
    plt.close()
    
    # Spectrogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, files, title in [(axes[0], drone_files[:1], 'Drone'), (axes[1], noise_files[:1], 'Noise')]:
        if files:
            sr, data = wavfile.read(files[0])
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            data = data / (np.max(np.abs(data)) + 1e-6)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=WINDOW_SIZE)), ref=np.max)
            librosa.display.specshow(D, sr=TARGET_SR, x_axis='time', y_axis='linear', ax=ax, cmap='magma')
            ax.set_title(title)
            ax.set_ylim(0, 4000)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'spectrogram_comparison.png'), dpi=150)
    plt.close()
    
    print(f"✅ Analysis complete for {subset_name}. Results saved to {analysis_dir}")
    return stats

# ==============================================================================
# 3. PROCESS ALL SUBSETS
# ==============================================================================
def main():
    print("🚀 Starting Acoustic Threat Bundle Analysis")
    print("="*70)
    
    subset_dirs = [d for d in glob.glob(os.path.join(SUBSETS_ROOT, '*')) if os.path.isdir(d)]
    subset_names = [os.path.basename(d) for d in subset_dirs]
    
    all_stats = []
    for subset_dir, subset_name in zip(subset_dirs, subset_names):
        try:
            stats = analyze_subset(subset_dir, subset_name)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error analyzing {subset_name}: {e}")
    
    # Full training set
    try:
        full_train_stats = analyze_subset(
            os.path.join(DRONE_ANALYSIS_ROOT, 'train'),
            'full_train'
        )
        if full_train_stats:
            all_stats.append(full_train_stats)
    except Exception as e:
        print(f"❌ Error analyzing full_train: {e}")
    
    # Save master summary
    master_summary = {
        'subsets': all_stats,
        'feature_names': ['ZCR', 'RMS', 'Centroid', 'Rolloff'] + [f'Band_{i+1}' for i in range(NUM_HARMONICS)] + [f'MFCC_{i+1}' for i in range(NUM_MFCC)]
    }
    with open(os.path.join(MASTER_ANALYSIS_DIR, 'master_summary.json'), 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    # Create Threat Bundle Config
    if all_stats:
        full_stats = next((s for s in all_stats if s['subset_name'] == 'full_train'), all_stats[-1])
        threat_bundle = {
            'harmonic_bands_hz': full_stats['harmonic_bands'],
            'band_thresholds_90pct': full_stats['band_thresholds_90pct'],
            'knowledge_bonus_bands': full_stats['harmonic_bands'],
            'knowledge_bonus_thresholds': full_stats['band_thresholds_90pct']
        }
        with open(os.path.join(MASTER_ANALYSIS_DIR, 'threat_bundle_config.json'), 'w') as f:
            json.dump(threat_bundle, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ MASTER ANALYSIS COMPLETE")
    print(f"📁 Results saved to: {MASTER_ANALYSIS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()