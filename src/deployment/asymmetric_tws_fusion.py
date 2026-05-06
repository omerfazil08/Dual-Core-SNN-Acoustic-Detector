# ==============================================================================
# FINAL DEPLOYMENT FUSION SWEEP (Raw Audio / Asymmetric Bebop Priority)
# ==============================================================================

import os, glob
import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal as scipy_signal
from numpy.lib.stride_tricks import sliding_window_view

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 FINAL ASYMMETRIC FUSION SWEEP ON: {device}")

# ==============================================================================
# 1. CONFIGURATION & FILE PATHS
# ==============================================================================
class Config:
    BEBOP_FOLDER = "/content/drone_audio_detector/data/DroneAudioDataset-master/Multiclass_Drone_Audio/bebop_1"
    MEMBO_FOLDER = "/content/drone_audio_detector/data/DroneAudioDataset-master/Multiclass_Drone_Audio/membo_1"
    NOISE_FOLDER = "/content/drone_audio_detector/data/DroneAudioDataset-master/Binary_Drone_Audio/unknown"
    
    # ⚠️ PATHS: Bebop is in Colab session, Membo is in Drive
    BEBOP_WEIGHTS_DIR = "/content/" 
    MEMBO_WEIGHTS_DIR = "/content/drive/MyDrive/" 

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 248
    HIDDEN_SIZE = 64
    TRAIN_RESERVE = 200 

# ==============================================================================
# 2. STANDARD DATA LOADER (No Digital Earplugs)
# ==============================================================================
def load_and_quantize(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1: data = np.mean(data, axis=1)
    
    if np.max(np.abs(data)) > 0: data = data.astype(np.float32) / np.max(np.abs(data))
    if sr != Config.SAMPLE_RATE:
        num_target = int(len(data) * Config.SAMPLE_RATE / sr)
        data = scipy_signal.resample(data, num_target)
    return np.clip(data * 127.0, -127, 127).astype(np.int32)

def build_massive_fusion_set():
    np.random.seed(42) 
    bebop_files = sorted(glob.glob(os.path.join(Config.BEBOP_FOLDER, "*.wav")))
    membo_files = sorted(glob.glob(os.path.join(Config.MEMBO_FOLDER, "*.wav")))
    noise_files = sorted(glob.glob(os.path.join(Config.NOISE_FOLDER, "*.wav")))
    
    np.random.shuffle(bebop_files); np.random.shuffle(membo_files); np.random.shuffle(noise_files)

    test_bebop = bebop_files[Config.TRAIN_RESERVE:]
    test_membo = membo_files[Config.TRAIN_RESERVE:]
    test_noise = noise_files[Config.TRAIN_RESERVE:]

    test_files = [(f,1) for f in test_bebop] + [(f,2) for f in test_membo] + [(f,0) for f in test_noise]
    print(f"⏳ Loading Unified Dataset: {len(test_bebop)} Bebop | {len(test_membo)} Membo | {len(test_noise)} Noise...")

    windows, labels = [], []
    for f, lbl in test_files:
        try:
            q = load_and_quantize(f)
            n_win = len(q) // Config.WINDOW_SIZE
            for i in range(n_win):
                windows.append(q[i*Config.WINDOW_SIZE:(i+1)*Config.WINDOW_SIZE])
                labels.append(lbl)
        except: pass

    X_test = np.stack(windows)
    Y_test = np.array(labels, dtype=np.int32)
    print(f"✅ Data Rebuilt: {len(X_test)} raw windows (15.5ms each)")
    return X_test, Y_test

# ==============================================================================
# 3. SNN HARDWARE SIMULATOR
# ==============================================================================
def load_core(prefix, directory):
    w1 = np.load(os.path.join(directory, f"best_{prefix}_weights_w1.npy"))
    w2 = np.load(os.path.join(directory, f"best_{prefix}_weights_w2.npy"))
    g = np.load(os.path.join(directory, f"best_{prefix}_genome_finetuned.npy"))
    return w1, w2, g[:64], g[64], g[65]

def simulate_core(X_data, w1, w2, leaks, t1, t2):
    POP, SAMPLES, _ = X_data.shape
    H = Config.HIDDEN_SIZE
    
    W1 = torch.tensor(w1, dtype=torch.int32, device=device).view(1, 1, H)
    W2 = torch.tensor(w2, dtype=torch.int32, device=device).view(1, 1, H)
    L  = torch.tensor(leaks, dtype=torch.int32, device=device).view(1, 1, H)
    T1 = torch.tensor([t1], dtype=torch.int32, device=device).view(1, 1, 1)
    T2 = torch.tensor([t2], dtype=torch.int32, device=device).view(1, 1)

    mem1 = torch.zeros((POP, SAMPLES, H), dtype=torch.int32, device=device)
    mem2 = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)
    alarms = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)

    for step in range(Config.WINDOW_SIZE):
        x_t = torch.abs(X_data[:, :, step]).unsqueeze(2)
        cur1 = x_t * W1
        mem1 = (mem1 >> L) + cur1
        spk1 = (mem1 > T1).to(torch.int32)
        mem1 = mem1 * (1 - spk1)

        cur2 = torch.sum(spk1 * W2, dim=2)
        mem2 = (mem2 >> 1) + cur2
        spk2 = (mem2 > T2).to(torch.int32)
        mem2 = mem2 * (1 - spk2)
        alarms |= spk2
    return alarms

# ==============================================================================
# 4. ASYMMETRIC PARETO SWEEP ENGINE
# ==============================================================================
if __name__ == "__main__":
    X_test, Y_test = build_massive_fusion_set()
    X_test_t = torch.tensor(X_test, dtype=torch.int32, device=device).unsqueeze(0)

    try:
        print("\n⚡ Powering up Core A (Bebop)...")
        b_w1, b_w2, b_L, b_T1, b_T2 = load_core("bebop", Config.BEBOP_WEIGHTS_DIR)
        with torch.no_grad():
            preds_B = simulate_core(X_test_t, b_w1, b_w2, b_L, b_T1, b_T2)[0].cpu().numpy()

        print("⚡ Powering up Core B (Membo)...")
        m_w1, m_w2, m_L, m_T1, m_T2 = load_core("mambo", Config.MEMBO_WEIGHTS_DIR)
        with torch.no_grad():
            preds_M = simulate_core(X_test_t, m_w1, m_w2, m_L, m_T1, m_T2)[0].cpu().numpy()
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing SNN .npy files. Double check your directory paths in the Config!")
        print(e)
        exit()

    print("\n" + "=" * 135)
    print("🚀 INITIATING ASYMMETRIC DUAL-CORE PARETO SWEEP (BEBOP PRIORITY)")
    print("=" * 135)

    print(f"{'Size':>4} | "
          f"{'M1 B-Acc':>8} | {'RecB':>6} | {'RecM':>6} | {'SpecN':>6} | {'(Tb, Tm)':>9} || "
          f"{'M3 B-Acc':>8} | {'RecB':>6} | {'RecM':>6} | {'SpecN':>6} | {'(Tb, Tm)':>9} | {'Slides'}")
    print("-" * 135)

    best_overall_m1 = {"b_acc": 0}
    best_overall_m3 = {"b_acc": 0}

    for m_size in range(1, 65):
        n_mac = len(preds_B) // m_size
        if n_mac < 6: continue 

        sums_b = np.array([preds_B[i*m_size:(i+1)*m_size].sum() for i in range(n_mac)])
        sums_m = np.array([preds_M[i*m_size:(i+1)*m_size].sum() for i in range(n_mac)])
        y_true = np.array([np.bincount(Y_test[i*m_size:(i+1)*m_size], minlength=3).argmax() for i in range(n_mac)])

        total_b = np.sum(y_true == 1) + 1e-6
        total_m = np.sum(y_true == 2) + 1e-6
        total_n = np.sum(y_true == 0) + 1e-6

        best_m1 = {"b_acc": 0, "rec_b": 0, "rec_m": 0, "spec_n": 0, "Tb": 0, "Tm": 0}
        best_m3 = {"b_acc": 0, "rec_b": 0, "rec_m": 0, "spec_n": 0, "Tb": 0, "Tm": 0, "W": 0}

        # ── VECTORIZED 2D THRESHOLD SWEEP ──
        for Tb in range(0, m_size + 1):
            alarm_b = sums_b >= Tb
            for Tm in range(0, m_size + 1):
                alarm_m = sums_m >= Tm

                # --- METHOD 1 LOGIC (Single Window) ---
                pred_m1 = np.zeros_like(y_true)
                
                # ASYMMETRIC LOGIC: Bebop strictly overwrites Membo
                pred_m1[alarm_m] = 2 
                pred_m1[alarm_b] = 1

                rec_b_m1 = np.sum((pred_m1 == 1) & (y_true == 1)) / total_b
                rec_m_m1 = np.sum((pred_m1 == 2) & (y_true == 2)) / total_m
                spec_n_m1 = np.sum((pred_m1 == 0) & (y_true == 0)) / total_n
                
                b_acc_m1 = (rec_b_m1 + rec_m_m1 + spec_n_m1) / 3.0

                if b_acc_m1 > best_m1["b_acc"]:
                    best_m1 = {"b_acc": b_acc_m1, "rec_b": rec_b_m1, "rec_m": rec_m_m1, "spec_n": spec_n_m1, "Tb": Tb, "Tm": Tm}

                # --- METHOD 3 LOGIC (Adjustable Slides W) ---
                for W in [2, 3, 4, 5]:
                    if n_mac < W: continue
                    
                    alarm_b_w = sliding_window_view(alarm_b, window_shape=W).all(axis=1)
                    alarm_m_w = sliding_window_view(alarm_m, window_shape=W).all(axis=1)
                    
                    y_true_w = y_true[W-1:]

                    pred_w = np.zeros_like(y_true_w)
                    
                    # ASYMMETRIC LOGIC: Bebop strictly overwrites Membo
                    pred_w[alarm_m_w] = 2 
                    pred_w[alarm_b_w] = 1

                    tot_b_w = np.sum(y_true_w == 1) + 1e-6
                    tot_m_w = np.sum(y_true_w == 2) + 1e-6
                    tot_n_w = np.sum(y_true_w == 0) + 1e-6

                    rec_b_w = np.sum((pred_w == 1) & (y_true_w == 1)) / tot_b_w
                    rec_m_w = np.sum((pred_w == 2) & (y_true_w == 2)) / tot_m_w
                    spec_n_w = np.sum((pred_w == 0) & (y_true_w == 0)) / tot_n_w
                    
                    b_acc_w = (rec_b_w + rec_m_w + spec_n_w) / 3.0

                    if b_acc_w > best_m3["b_acc"]:
                        best_m3 = {"b_acc": b_acc_w, "rec_b": rec_b_w, "rec_m": rec_m_w, "spec_n": spec_n_w, "Tb": Tb, "Tm": Tm, "W": W}

        if best_m1["b_acc"] > best_overall_m1["b_acc"]: best_overall_m1 = dict(best_m1, size=m_size)
        if best_m3["b_acc"] > best_overall_m3["b_acc"]: best_overall_m3 = dict(best_m3, size=m_size)

        print(f"{m_size:>4} | "
              f"{best_m1['b_acc']*100:>7.1f}% | {best_m1['rec_b']*100:>5.1f}% | {best_m1['rec_m']*100:>5.1f}% | {best_m1['spec_n']*100:>5.1f}% | "
              f"({best_m1['Tb']:>2},{best_m1['Tm']:>2}) || "
              f"{best_m3['b_acc']*100:>7.1f}% | {best_m3['rec_b']*100:>5.1f}% | {best_m3['rec_m']*100:>5.1f}% | {best_m3['spec_n']*100:>5.1f}% | "
              f"({best_m3['Tb']:>2},{best_m3['Tm']:>2}) | W={best_m3['W']}")

    print("\n" + "=" * 135)
    print("🏆 ULTIMATE MULTI-CLASS CHAMPIONS (BEBOP PRIORITY OVERRIDE)")
    print("=" * 135)
    print("Method 1 Champion (Single Window):")
    print(f"   Macro Size: {best_overall_m1.get('size', 'N/A')} | T_bebop >= {best_overall_m1.get('Tb', 0)} | T_membo >= {best_overall_m1.get('Tm', 0)}")
    print(f"   Metrics -> Bebop Recall: {best_overall_m1.get('rec_b', 0)*100:.1f}% | Membo Recall: {best_overall_m1.get('rec_m', 0)*100:.1f}% | Noise Spec: {best_overall_m1.get('spec_n', 0)*100:.1f}%")
    print(f"   Final 3-Class Balanced Accuracy: {best_overall_m1.get('b_acc', 0)*100:.2f}%\n")

    print("Method 3 Champion (Adjustable Slide Coincidence):")
    print(f"   Macro Size: {best_overall_m3.get('size', 'N/A')} | T_bebop >= {best_overall_m3.get('Tb', 0)} | T_membo >= {best_overall_m3.get('Tm', 0)} | Slides (W) = {best_overall_m3.get('W', 0)}")
    print(f"   Metrics -> Bebop Recall: {best_overall_m3.get('rec_b', 0)*100:.1f}% | Membo Recall: {best_overall_m3.get('rec_m', 0)*100:.1f}% | Noise Spec: {best_overall_m3.get('spec_n', 0)*100:.1f}%")
    print(f"   Final 3-Class Balanced Accuracy: {best_overall_m3.get('b_acc', 0)*100:.2f}%")