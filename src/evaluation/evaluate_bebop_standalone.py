# ==============================================================================
# STANDALONE VHDL MACRO-WINDOW EVALUATOR (64-Neuron / 248-Window Architecture)
# ==============================================================================

import os, glob
import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal as scipy_signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ==============================================================================
# 1. CONFIGURATION & DATA RE-LOADER
# ==============================================================================
class Config:
    DRONE_FOLDER = "/content/drone_audio_detector/data/DroneAudioDataset-master/Multiclass_Drone_Audio/bebop_1"
    NOISE_FOLDER = "/content/drone_audio_detector/data/DroneAudioDataset-master/Binary_Drone_Audio/unknown"
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 248
    HIDDEN_SIZE = 64
    FILES_PER_CLASS = 200
    TRAIN_RATIO = 0.80

def load_and_quantize(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1: data = np.mean(data, axis=1)
    if np.max(np.abs(data)) > 0: data = data.astype(np.float32) / np.max(np.abs(data))
    if sr != Config.SAMPLE_RATE:
        num_target = int(len(data) * Config.SAMPLE_RATE / sr)
        data = scipy_signal.resample(data, num_target)
    return np.clip(data * 127.0, -127, 127).astype(np.int32)

def build_test_set():
    np.random.seed(42) # Set seed for consistent evaluation
    drone_files = sorted(glob.glob(os.path.join(Config.DRONE_FOLDER, "*.wav")))
    noise_files = sorted(glob.glob(os.path.join(Config.NOISE_FOLDER, "*.wav")))
    np.random.shuffle(drone_files); np.random.shuffle(noise_files)

    drone_files = drone_files[:Config.FILES_PER_CLASS]
    noise_files = noise_files[:Config.FILES_PER_CLASS]
    split = int(Config.FILES_PER_CLASS * Config.TRAIN_RATIO)

    test_files  = [(f,1) for f in drone_files[split:]] + [(f,0) for f in noise_files[split:]]

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
    print(f"✅ Test Data Rebuilt: {len(X_test)} windows (15.5ms each)")
    return X_test, Y_test

# ==============================================================================
# 2. SNN HARDWARE SIMULATOR
# ==============================================================================
def simulate_population(X_raw, W1, W2, LEAKS, T1, T2):
    POP, SAMPLES, _ = X_raw.shape
    H = Config.HIDDEN_SIZE
    mem1 = torch.zeros((POP, SAMPLES, H), dtype=torch.int32, device=device)
    mem2 = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)
    alarms = torch.zeros((POP, SAMPLES), dtype=torch.int32, device=device)

    w1 = W1.view(POP, 1, H)
    w2 = W2.view(POP, 1, H)
    leaks = LEAKS.view(POP, 1, H)
    t1 = T1.view(POP, 1, 1)
    t2 = T2.view(POP, 1)

    for step in range(Config.WINDOW_SIZE):
        x_t = torch.abs(X_raw[:, :, step]).unsqueeze(2)
        cur1 = x_t * w1
        mem1 = (mem1 >> leaks) + cur1
        spk1 = (mem1 > t1).to(torch.int32)
        mem1 = mem1 * (1 - spk1)

        cur2 = torch.sum(spk1 * w2, dim=2)
        mem2 = (mem2 >> 1) + cur2
        spk2 = (mem2 > t2).to(torch.int32)
        mem2 = mem2 * (1 - spk2)
        alarms |= spk2
    return alarms
# ==============================================================================
# 3. EXECUTION & EVALUATION
# ==============================================================================
if __name__ == "__main__":
    X_test, Y_test = build_test_set()

    # Load your 3 saved files
    try:
        w1 = np.load("/content/best_bebop_weights_w1 (6).npy")
        w2 = np.load("/content/best_bebop_weights_w2 (5).npy")
        discrete_genome = np.load("/content/best_bebop_genome_finetuned (6).npy")
    except FileNotFoundError:
        print("❌ ERROR: Could not find the .npy files. Make sure they are uploaded to /content/!")
        exit()

    # Stitch into final master array (194 elements)
    master_genome = np.concatenate([w1, w2, discrete_genome]).astype(np.int32)
    np.save("/content/master_bebop_genome_64_neuron_1.npy", master_genome)
    print("✅ 64-Neuron Master Genome stitched and saved!")

    # Decode for PyTorch simulation
    W1_t = torch.tensor(w1, dtype=torch.int32, device=device).view(1, -1)
    W2_t = torch.tensor(w2, dtype=torch.int32, device=device).view(1, -1)
    L_t  = torch.tensor(discrete_genome[:64], dtype=torch.int32, device=device).view(1, -1)
    T1_t = torch.tensor([discrete_genome[64]], dtype=torch.int32, device=device).view(1, 1)
    T2_t = torch.tensor([discrete_genome[65]], dtype=torch.int32, device=device)

    # Run Micro-Inference
    X_test_t = torch.tensor(X_test, dtype=torch.int32, device=device).unsqueeze(0)
    with torch.no_grad():
        micro_preds = simulate_population(X_test_t, W1_t, W2_t, L_t, T1_t, T2_t)[0].cpu().numpy()

    TP_m = np.sum((micro_preds==1) & (Y_test==1))
    TN_m = np.sum((micro_preds==0) & (Y_test==0))
    FP_m = np.sum((micro_preds==1) & (Y_test==0))
    FN_m = np.sum((micro_preds==0) & (Y_test==1))
    micro_acc = (TP_m+TN_m)/(TP_m+TN_m+FP_m+FN_m+1e-6)
    print(f"\n🎯 Baseline Micro-Accuracy: {micro_acc*100:.2f}%\n")

    # ==========================================================================
    # DYNAMIC MACRO-WINDOW EVALUATOR
    # ==========================================================================
    MACRO_SIZE = 64  # <--- CHANGE THIS TO TEST LATENCY (e.g., 48, 32, 16)

    num_macros = len(micro_preds) // MACRO_SIZE
    macro_sums = np.array([micro_preds[i*MACRO_SIZE:(i+1)*MACRO_SIZE].sum() for i in range(num_macros)])
    macro_labels = np.array([1 if Y_test[i*MACRO_SIZE:(i+1)*MACRO_SIZE].mean() > 0.5 else 0 for i in range(num_macros)])

    print("=" * 70)
    print(f"bebop MACRO‑WINDOW EVALUATION ({MACRO_SIZE}-Frame Window)")
    print(f"Total Windows: {len(macro_sums)} ({macro_labels.sum()} drone, {(1-macro_labels).sum()} noise)")
    print("=" * 70)

    # --- TRACKERS FOR AUTOMATIC JSON ---
    best_m2 = {"rec": 0, "far": 100, "str": "None"}
    best_m3_god = {"rec": 0, "far": 100, "str": "None"}
    best_m3_ghost = {"rec": 0, "far": 100, "str": "None"}

    # ── METHOD 1: Simple Single Threshold ──
    print("\n--- METHOD 1: SIMPLE SINGLE THRESHOLD ---")
    print(f"{'T':>5} | {'Recall':>7} | {'FAR':>7} | {'Acc':>7}")
    print("-" * 38)
    # Dynamic range: Up to MACRO_SIZE total spikes
    for T in range(0, MACRO_SIZE + 1, 1):
        TP = TN = FP = FN = 0
        for s, lbl in zip(macro_sums, macro_labels):
            alarm = 1 if s >= T else 0
            if alarm and lbl: TP += 1
            elif not alarm and not lbl: TN += 1
            elif alarm and not lbl: FP += 1
            else: FN += 1
        rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100
        print(f"{T:5d} | {rec:6.1f}% | {far:6.1f}% | {acc:6.1f}%")

    # ── METHOD 2: Forward‑Looking Reflex ──
    print("\n--- METHOD 2: FORWARD‑LOOKING REFLEX ---")
    print(f"{'X/Y':>8} | {'Recall':>7} | {'FAR':>7} | {'Acc':>7}")
    print("-" * 48)
    for X in range(0, MACRO_SIZE):
        for Y in range(X+1, MACRO_SIZE + 1):
            TP = TN = FP = FN = 0
            for i in range(num_macros - 1):
                s = macro_sums[i]
                lbl = macro_labels[i]
                if s >= Y: alarm = 1
                elif s >= X: alarm = 1 if macro_sums[i+1] >= X else 0
                else: alarm = 0

                if alarm and lbl: TP += 1
                elif not alarm and not lbl: TN += 1
                elif alarm and not lbl: FP += 1
                else: FN += 1
            rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100

            if far <= 15.0:
                print(f"  ≥{X:<2d} ≥{Y:<2d} | {rec:6.1f}% | {far:6.1f}% | {acc:6.1f}%")
                # Auto-Tracker logic for Method 2 (Target: Max Recall with FAR <= 5%)
                if far <= 5.0 and rec >= best_m2["rec"]:
                    best_m2 = {"rec": rec, "far": far, "str": f"X>={X}, Y>={Y} | Recall: {rec:.1f}% | FAR: {far:.1f}%"}

    # ── METHOD 3: Coincidence ──
    print("\n--- METHOD 3: 3‑WINDOW COINCIDENCE (M‑OF‑N) ---")
    print(f"{'T≥':>5} | {'M/N':>5} | {'Recall':>7} | {'FAR':>7} | {'Acc':>7}")
    print("-" * 48)
    for T_high in range(0, MACRO_SIZE + 1):
        for M in [2, 3]:
            N = 3
            TP = TN = FP = FN = 0
            for i in range(num_macros - N + 1):
                block = macro_sums[i:i+N]
                lbl = macro_labels[i]
                high_count = (block >= T_high).sum()
                alarm = 1 if high_count >= M else 0

                if alarm and lbl: TP += 1
                elif not alarm and not lbl: TN += 1
                elif alarm and not lbl: FP += 1
                else: FN += 1
            rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100

            if far <= 15.0:
                print(f"{T_high:5d} | {M}/{N:3d} | {rec:6.1f}% | {far:6.1f}% | {acc:6.1f}%")

                # Auto-Tracker for God Mode (Target: Max Recall with FAR <= 6%)
                if far <= 6.0:
                    if rec > best_m3_god["rec"] or (rec == best_m3_god["rec"] and far < best_m3_god["far"]):
                        best_m3_god = {"rec": rec, "far": far, "str": f"T>={T_high}, {M}/{N} | Recall: {rec:.1f}% | FAR: {far:.1f}%"}

                # Auto-Tracker for Ghost Mode (Target: FAR exactly 0.0%, Max Recall)
                if far == 0.0 and rec > best_m3_ghost["rec"]:
                    best_m3_ghost = {"rec": rec, "far": far, "str": f"T>={T_high}, {M}/{N} | Recall: {rec:.1f}% | FAR: {far:.1f}%"}

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    # ==========================================================================
    # AUTO-JSON GENERATOR
    # ==========================================================================
    import json

    telemetry = {
        "drone_type": "bebop",
        "architecture": {
            "window_size_samples": Config.WINDOW_SIZE,
            "hidden_neurons": Config.HIDDEN_SIZE,
            "macro_size_frames": MACRO_SIZE
        },
        "micro_accuracy": f"{micro_acc*100:.2f}%",
        "best_vhdl_deployments": {
            "method_2_fast": best_m2["str"],
            "method_3_god_mode": best_m3_god["str"],
            "method_3_ghost_mode": best_m3_ghost["str"]
        }
    }

    # Save it alongside the .npy file with the macro size in the name
    json_filename = f"/content/bebop_genome_{Config.HIDDEN_SIZE}N_macro{MACRO_SIZE}_stats_1.json"
    with open(json_filename, "w") as f:
        json.dump(telemetry, f, indent=4)
    print(f"✅ Telemetry saved to {json_filename}")
    print(json.dumps(telemetry, indent=4))
# ==========================================================================
    # 4. THE PARETO FRONTIER SWEEP (MACRO_SIZE 1 to 64)
    # ==========================================================================
    print("\n" + "=" * 115)
    print("🚀 INITIATING MACRO-SIZE PARETO FRONTIER SWEEP (Reaction Time vs. Reliability)")
    print("=" * 115)

    # Header for the massive table
    print(f"{'Size':>4} | "
          f"{'M1 BestAcc':>10} | {'M1 MinFAR@100':>13} | {'M1 MaxRec@0':>11} || "
          f"{'M2 BestAcc':>10} | {'M2 MinFAR@100':>13} | {'M2 MaxRec@0':>11} || "
          f"{'M3 BestAcc':>10} | {'M3 MinFAR@100':>13} | {'M3 MaxRec@0':>11}")
    print("-" * 115)

    for m_size in range(1, 65):
        # 1. Rebuild arrays for this specific macro size
        n_mac = len(micro_preds) // m_size
        if n_mac < 5: continue # Skip if data is too small to evaluate

        m_sums = np.array([micro_preds[i*m_size:(i+1)*m_size].sum() for i in range(n_mac)])
        m_labels = np.array([1 if Y_test[i*m_size:(i+1)*m_size].mean() > 0.5 else 0 for i in range(n_mac)])

        # Trackers: [Best Acc, Min FAR @ 100 Recall, Max Recall @ 0 FAR]
        t1 = [0.0, 100.0, 0.0]
        t2 = [0.0, 100.0, 0.0]
        t3 = [0.0, 100.0, 0.0]

        # ── METHOD 1 SWEEP ──
        for T in range(0, m_size + 1):
            TP = TN = FP = FN = 0
            for s, lbl in zip(m_sums, m_labels):
                alarm = 1 if s >= T else 0
                if alarm and lbl: TP += 1
                elif not alarm and not lbl: TN += 1
                elif alarm and not lbl: FP += 1
                else: FN += 1
            rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100

            if acc > t1[0]: t1[0] = acc
            if rec >= 99.9 and far < t1[1]: t1[1] = far
            if far <= 0.01 and rec > t1[2]: t1[2] = rec

        # ── METHOD 2 SWEEP ──
        for X in range(0, m_size):
            for Y in range(X+1, m_size + 1):
                TP = TN = FP = FN = 0
                for i in range(n_mac - 1):
                    s, lbl = m_sums[i], m_labels[i]
                    if s >= Y: alarm = 1
                    elif s >= X: alarm = 1 if m_sums[i+1] >= X else 0
                    else: alarm = 0
                    if alarm and lbl: TP += 1
                    elif not alarm and not lbl: TN += 1
                    elif alarm and not lbl: FP += 1
                    else: FN += 1
                rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100

                if acc > t2[0]: t2[0] = acc
                if rec >= 99.9 and far < t2[1]: t2[1] = far
                if far <= 0.01 and rec > t2[2]: t2[2] = rec

        # ── METHOD 3 SWEEP ──
        for T_high in range(0, m_size + 1):
            for M in [2, 3]:
                N = 3
                TP = TN = FP = FN = 0
                for i in range(n_mac - N + 1):
                    lbl = m_labels[i]
                    alarm = 1 if (m_sums[i:i+N] >= T_high).sum() >= M else 0
                    if alarm and lbl: TP += 1
                    elif not alarm and not lbl: TN += 1
                    elif alarm and not lbl: FP += 1
                    else: FN += 1
                rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100

                if acc > t3[0]: t3[0] = acc
                if rec >= 99.9 and far < t3[1]: t3[1] = far
                if far <= 0.01 and rec > t3[2]: t3[2] = rec

        # Print the row for this Macro Size
        print(f"{m_size:>4} | "
              f"{t1[0]:>9.1f}% | {t1[1]:>12.1f}% | {t1[2]:>10.1f}% || "
              f"{t2[0]:>9.1f}% | {t2[1]:>12.1f}% | {t2[2]:>10.1f}% || "
              f"{t3[0]:>9.1f}% | {t3[1]:>12.1f}% | {t3[2]:>10.1f}%")
# ==========================================================================
    # 5. THE CHAMPION SELECTOR & JSON GENERATOR
    # ==========================================================================
    print("\n" + "=" * 115)
    print("🏆 EXTRACTING OPTIMAL VHDL DEPLOYMENT ARCHITECTURES")
    print("=" * 115)

    # Dictionaries to store the ultimate champions
    champs_m1 = {"best_acc": None, "min_far": None, "max_rec": None, "best_efficiency": None}
    champs_m2 = {"best_acc": None, "min_far": None, "max_rec": None, "best_efficiency": None}
    champs_m3 = {"best_acc": None, "min_far": None, "max_rec": None, "best_efficiency": None}

    # Tracking variables for the sweep
    max_acc_m1 = max_acc_m2 = max_acc_m3 = 0.0
    min_far_m1 = min_far_m2 = min_far_m3 = 100.0
    max_rec_m1 = max_rec_m2 = max_rec_m3 = 0.0
    best_eff_m1 = best_eff_m2 = best_eff_m3 = 0.0

    LATENCY_PENALTY = 0.2 # Penalty per frame delayed (adjustable)

    # We re-run the sweep logic (silently) but this time we track the actual thresholds
    for m_size in range(1, 65):
        n_mac = len(micro_preds) // m_size
        if n_mac < 5: continue
        m_sums = np.array([micro_preds[i*m_size:(i+1)*m_size].sum() for i in range(n_mac)])
        m_labels = np.array([1 if Y_test[i*m_size:(i+1)*m_size].mean() > 0.5 else 0 for i in range(n_mac)])

        # ── METHOD 1 SCAN ──
        for T in range(0, m_size + 1):
            TP = TN = FP = FN = 0
            for s, lbl in zip(m_sums, m_labels):
                alarm = 1 if s >= T else 0
                if alarm and lbl: TP += 1
                elif not alarm and not lbl: TN += 1
                elif alarm and not lbl: FP += 1
                else: FN += 1
            rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100
            eff = acc - (m_size * LATENCY_PENALTY)

            stat_str = f"Size: {m_size} | T>={T} | Rec: {rec:.1f}% | FAR: {far:.1f}% | Acc: {acc:.1f}%"
            if acc > max_acc_m1: max_acc_m1 = acc; champs_m1["best_acc"] = stat_str
            if rec >= 99.9 and far < min_far_m1: min_far_m1 = far; champs_m1["min_far"] = stat_str
            if far <= 0.01 and rec > max_rec_m1: max_rec_m1 = rec; champs_m1["max_rec"] = stat_str
            if eff > best_eff_m1 and far < 10.0: best_eff_m1 = eff; champs_m1["best_efficiency"] = stat_str

        # ── METHOD 2 SCAN ──
        for X in range(0, m_size):
            for Y in range(X+1, m_size + 1):
                TP = TN = FP = FN = 0
                for i in range(n_mac - 1):
                    s, lbl = m_sums[i], m_labels[i]
                    if s >= Y: alarm = 1
                    elif s >= X: alarm = 1 if m_sums[i+1] >= X else 0
                    else: alarm = 0
                    if alarm and lbl: TP += 1
                    elif not alarm and not lbl: TN += 1
                    elif alarm and not lbl: FP += 1
                    else: FN += 1
                rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100
                eff = acc - (m_size * LATENCY_PENALTY)

                stat_str = f"Size: {m_size} | X>={X}, Y>={Y} | Rec: {rec:.1f}% | FAR: {far:.1f}% | Acc: {acc:.1f}%"
                if acc > max_acc_m2: max_acc_m2 = acc; champs_m2["best_acc"] = stat_str
                if rec >= 99.9 and far < min_far_m2: min_far_m2 = far; champs_m2["min_far"] = stat_str
                if far <= 0.01 and rec > max_rec_m2: max_rec_m2 = rec; champs_m2["max_rec"] = stat_str
                if eff > best_eff_m2 and far < 10.0: best_eff_m2 = eff; champs_m2["best_efficiency"] = stat_str

        # ── METHOD 3 SCAN ──
        for T_high in range(0, m_size + 1):
            for M in [2, 3]:
                N = 3
                TP = TN = FP = FN = 0
                for i in range(n_mac - N + 1):
                    lbl = m_labels[i]
                    alarm = 1 if (m_sums[i:i+N] >= T_high).sum() >= M else 0
                    if alarm and lbl: TP += 1
                    elif not alarm and not lbl: TN += 1
                    elif alarm and not lbl: FP += 1
                    else: FN += 1
                rec = TP/(TP+FN+1e-6)*100; far = FP/(FP+TN+1e-6)*100; acc = (TP+TN)/(TP+TN+FP+FN+1e-6)*100
                eff = acc - (m_size * LATENCY_PENALTY)

                stat_str = f"Size: {m_size} | T>={T_high}, {M}/{N} | Rec: {rec:.1f}% | FAR: {far:.1f}% | Acc: {acc:.1f}%"
                if acc > max_acc_m3: max_acc_m3 = acc; champs_m3["best_acc"] = stat_str
                if rec >= 99.9 and far < min_far_m3: min_far_m3 = far; champs_m3["min_far"] = stat_str
                if far <= 0.01 and rec > max_rec_m3: max_rec_m3 = rec; champs_m3["max_rec"] = stat_str
                if eff > best_eff_m3 and far < 10.0: best_eff_m3 = eff; champs_m3["best_efficiency"] = stat_str

    print("\nMETHOD 1 (Simple Threshold) Champions:")
    for k, v in champs_m1.items(): print(f"  - {k:>15}: {v}")
    print("\nMETHOD 2 (Forward Reflex) Champions:")
    for k, v in champs_m2.items(): print(f"  - {k:>15}: {v}")
    print("\nMETHOD 3 (M-of-N Coincidence) Champions:")
    for k, v in champs_m3.items(): print(f"  - {k:>15}: {v}")

    # ==========================================================================
    # AUTO-JSON GENERATOR
    # ==========================================================================
    import json

    telemetry = {
        "drone_type": "bebop",
        "architecture": {
            "window_size_samples": Config.WINDOW_SIZE,
            "hidden_neurons": Config.HIDDEN_SIZE,
            "latency_penalty_weight": LATENCY_PENALTY
        },
        "micro_accuracy": f"{micro_acc*100:.2f}%",
        "vhdl_deployments": {
            "method_1": champs_m1,
            "method_2": champs_m2,
            "method_3": champs_m3
        }
    }

    json_filename = f"/content/bebop_genome_{Config.HIDDEN_SIZE}N_master_stats_1.json"
    with open(json_filename, "w") as f:
        json.dump(telemetry, f, indent=4)
    print(f"\n✅ Master Telemetry saved to {json_filename}")