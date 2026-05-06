# ==============================================================================
# MEMETIC HYBRID SNN TRAINING PIPELINE — MEMBO DRONE DETECTOR
# Phase 1 : Island-Model GA with Fisher Timbre Tutor (MFCC_11 + MFCC_12)
# Phase 2 : Surrogate-Gradient SGD with Straight-Through Estimator
# Topology : 1-32-1 LIF SNN (0-DSP, VHDL-accurate)
# AC Fix   : signed int8 data loader → abs() inside forward pass
# Author   : Ömer Fazıl Orhan — AGU SDP2
# ==============================================================================

import os
import gc
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy.io import wavfile
from scipy import signal as scipy_signal

# ------------------------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Computation Device: {device}")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # Signal
    WINDOW_SIZE  = 1024
    SAMPLE_RATE  = 16000

    # Architecture
    HIDDEN_SIZE   = 32
    MAX_WEIGHT    = 127
    MAX_LEAK_SHIFT = 3
    T1_MIN        = 50       # tighter range — Membo needs finer threshold
    T1_MAX        = 5000
    T2_MIN        = 10
    T2_MAX        = 110

    # Paths
    DRONE_TRAIN_DIR = '/content/drone_audio_detector/drone_analysis_data/membo_phase/train/membo/'
    NOISE_TRAIN_DIR = '/content/drone_audio_detector/drone_analysis_data/membo_phase/train/unknown/'
    TEST_DRONE_DIR  = '/content/drone_audio_detector/drone_analysis_data/membo_phase/test/membo/'
    TEST_NOISE_DIR  = '/content/drone_audio_detector/drone_analysis_data/membo_phase/test/unknown/'
    GENOME_SAVE     = '/content/best_memetic_genome.npy'

    # ── Phase 1 GA (Island Model) ──────────────────────────────────────────
    N_ISLANDS        = 10
    ISLAND_POP       = 30        # 10 × 30 = 300 total genomes
    GENERATIONS      = 100
    TOURNAMENT_SIZE  = 5
    CROSSOVER_RATE   = 0.80
    MUTATION_RATE    = 0.10
    MIGRATION_EVERY  = 25        # migrate best individual between islands
    MIGRATION_COUNT  = 2         # number of migrants per island per event
    ELITE_PER_ISLAND = 3

    # Stagnation
    STAGNATION_LIMIT        = 30
    STAGNATION_BOOST        = 0.25

    # Curriculum
    RAW_ONLY_GENERATIONS = 10
    BONUS_RAMP_END        = 60
    MAX_BONUS_WEIGHT      = 0.50

    # Fisher Timbre Tutor — Membo 2-sigma bounding box
    # MFCC_11 (librosa index 10) and MFCC_12 (librosa index 11)
    MFCC11_MIN =  -9.8
    MFCC11_MAX =  35.6
    MFCC12_MIN =  -6.8
    MFCC12_MAX =  35.4

    # ── Phase 2 SGD ────────────────────────────────────────────────────────
    SGD_EPOCHS      = 100
    SGD_LR          = 5e-3
    SGD_BATCH_SIZE  = 256
    EARLY_STOP_PAT  = 15         # patience epochs
    TARGET_SPIKES_DRONE = 12     # MSE target: drone window should fire 12 times
    TARGET_SPIKES_NOISE = 0      # MSE target: noise window should fire 0 times
    VAL_RATIO       = 0.20

    # Sub-batch for GA GPU evaluation
    SUB_BATCH = 100

# Genome layout:
#   [0      : HF]      W1    (32 × int8)
#   [HF     : 2HF]     W2    (32 × int8)
#   [2HF    : 3HF]     LEAKS (32 × int in 0..3)
#   [3HF]              T1    (global scalar)
#   [3HF+1]            T2    (global scalar)
HF        = Config.HIDDEN_SIZE
GENE_SIZE = HF * 3 + 2

# ==============================================================================
# 2. DATA PIPELINE  (AC Fix — signed int8, abs() applied inside forward pass)
# ==============================================================================
def load_and_quantize(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    if sr != Config.SAMPLE_RATE:
        n = int(len(data) * Config.SAMPLE_RATE / sr)
        data = scipy_signal.resample(data, n)
    # Signed 8-bit — preserve AC sign for MFCC teacher
    return np.clip(np.round(data * 127.0), -127, 127).astype(np.int8)


def files_to_windows(file_pairs):
    wins, labs = [], []
    for path, label in file_pairs:
        try:
            q = load_and_quantize(path)
            nw = len(q) // Config.WINDOW_SIZE
            for i in range(nw):
                wins.append(q[i*Config.WINDOW_SIZE:(i+1)*Config.WINDOW_SIZE])
                labs.append(label)
        except Exception:
            pass
    if not wins:
        return np.zeros((0, Config.WINDOW_SIZE), dtype=np.int8), np.zeros(0, dtype=np.int32)
    return np.stack(wins).astype(np.int8), np.array(labs, dtype=np.int32)


def build_datasets():
    drone_tr = sorted(glob.glob(os.path.join(Config.DRONE_TRAIN_DIR, '*.wav')))
    noise_tr = sorted(glob.glob(os.path.join(Config.NOISE_TRAIN_DIR, '*.wav')))
    drone_te = sorted(glob.glob(os.path.join(Config.TEST_DRONE_DIR,  '*.wav')))
    noise_te = sorted(glob.glob(os.path.join(Config.TEST_NOISE_DIR,  '*.wav')))

    n = min(len(drone_tr), len(noise_tr))
    np.random.shuffle(drone_tr); np.random.shuffle(noise_tr)
    drone_tr, noise_tr = drone_tr[:n], noise_tr[:n]

    print(f"📂 Training: {n} drone + {n} noise files")

    train_pairs = [(f, 1) for f in drone_tr] + [(f, 0) for f in noise_tr]
    test_pairs  = ([(f, 1) for f in drone_te] + [(f, 0) for f in noise_te])

    print("🔄 Building train windows...")
    X_all, Y_all = files_to_windows(train_pairs)
    print("🔄 Building test windows...")
    X_test, Y_test = files_to_windows(test_pairs)

    # Val split
    n_val = int(len(X_all) * Config.VAL_RATIO)
    idx   = np.random.permutation(len(X_all))
    X_train, Y_train = X_all[idx[n_val:]], Y_all[idx[n_val:]]
    X_val,   Y_val   = X_all[idx[:n_val]], Y_all[idx[:n_val]]

    print(f"✅ TRAIN: {len(X_train)} | VAL: {len(X_val)} | TEST: {len(X_test)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# ==============================================================================
# 3. FISHER TIMBRE TUTOR — MFCC_11 & MFCC_12 EXTRACTOR
#    AC Fix: decode signed int8 back to float AC waveform before computing MFCCs
# ==============================================================================
def extract_mfcc_features(windows_int8):
    """
    Returns (N, 2) array of [MFCC_11, MFCC_12] computed on the AC waveform.
    MFCC_11 = librosa 0-index 10, MFCC_12 = librosa 0-index 11.
    Uses torchaudio for GPU-friendliness; falls back gracefully.
    """
    N = len(windows_int8)
    features = np.zeros((N, 2), dtype=np.float32)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate  = Config.SAMPLE_RATE,
        n_mfcc       = 13,
        melkwargs    = {
            'n_fft'   : Config.WINDOW_SIZE,
            'hop_length': Config.WINDOW_SIZE,
            'n_mels'  : 40,
        }
    )

    for i in range(N):
        ac = torch.tensor(
            windows_int8[i].astype(np.float32) / 127.0
        ).unsqueeze(0)  # (1, 1024)

        mfccs = mfcc_transform(ac)  # (1, 13, frames)
        # Take mean across time frames (usually 1 frame for hop=window)
        mfccs_mean = mfccs[0, :, 0] if mfccs.shape[2] == 1 else mfccs[0, :, :].mean(dim=1)
        features[i, 0] = mfccs_mean[10].item()   # MFCC_11 (0-indexed = 10)
        features[i, 1] = mfccs_mean[11].item()   # MFCC_12 (0-indexed = 11)

    return features   # (N, 2)


def build_golden_mask(features_np):
    """
    Returns bool array (N,): True if window is inside the 2σ Membo bounding box.
    """
    m11 = features_np[:, 0]
    m12 = features_np[:, 1]
    return (
        (m11 >= Config.MFCC11_MIN) & (m11 <= Config.MFCC11_MAX) &
        (m12 >= Config.MFCC12_MIN) & (m12 <= Config.MFCC12_MAX)
    )

# ==============================================================================
# 4. VHDL-ACCURATE GA SIMULATOR  (PHASE 1)
# ==============================================================================
def simulate_population_ga(X_signed, W1, W2, LEAKS, T1, T2):
    """
    X_signed : (POP, N, WINDOW_SIZE) int32
    AC Fix   : torch.abs() applied per time step before membrane charging.
    """
    POP, N, _ = X_signed.shape

    mem1   = torch.zeros((POP, N, HF), dtype=torch.int32, device=device)
    mem2   = torch.zeros((POP, N),     dtype=torch.int32, device=device)
    alarms = torch.zeros((POP, N),     dtype=torch.int32, device=device)

    w1    = W1.view(POP, 1, HF)
    w2    = W2.view(POP, 1, HF)
    leaks = LEAKS.view(POP, 1, HF)
    t1    = T1.view(POP, 1, 1)    # Global T1 broadcasts to all 32 neurons
    t2    = T2.view(POP, 1)

    for step in range(Config.WINDOW_SIZE):
        x_t  = torch.abs(X_signed[:, :, step]).unsqueeze(2)   # AC Fix
        cur1 = x_t * w1
        mem1 = (mem1 >> leaks) + cur1
        spk1 = (mem1 > t1).to(torch.int32)
        mem1 = mem1 * (1 - spk1)

        cur2 = torch.sum(spk1 * w2, dim=2)
        mem2 = (mem2 >> 1) + cur2
        spk2 = (mem2 > t2).to(torch.int32)
        mem2 = mem2 * (1 - spk2)
        alarms |= spk2

    return alarms   # (POP, N)

# ==============================================================================
# 5. GA FITNESS FUNCTIONS
# ==============================================================================
def balanced_accuracy(preds, Y_true_t):
    Y_exp = Y_true_t.unsqueeze(0)
    TP = torch.sum((preds == 1) & (Y_exp == 1), dim=1).float()
    TN = torch.sum((preds == 0) & (Y_exp == 0), dim=1).float()
    FP = torch.sum((preds == 1) & (Y_exp == 0), dim=1).float()
    FN = torch.sum((preds == 0) & (Y_exp == 1), dim=1).float()
    return ((TP/(TP+FN+1e-6) + TN/(TN+FP+1e-6)) / 2.0)   # (POP,)


def timbre_bonus(preds, golden_mask_t):
    """
    Precision of spikes on Golden Mask windows (confirmed Membo aerodynamic texture).
    """
    gold_exp     = golden_mask_t.int().unsqueeze(0)   # (1, N)
    total_spikes = torch.sum(preds, dim=1).float()    # (POP,)
    gold_spikes  = torch.sum(preds * gold_exp, dim=1).float()
    return gold_spikes / (total_spikes + 1e-6)        # (POP,)

# ==============================================================================
# 6. GENOME UTILITIES
# ==============================================================================
def _halton(n, base):
    """Single Halton sequence in [0, 1)."""
    seq = np.zeros(n)
    num, den = 0, 1
    for i in range(n):
        x = den - num
        if x == 1:
            num, den = 1, den * base
        else:
            y = den // base
            while x <= y:
                y //= base
            num = (base + 1) * y - x
        seq[i] = num / den
    return seq


def quasi_random_genome():
    """
    Initialise one genome using a Hammersley-style quasi-random map
    to improve global coverage vs. pure random init.
    Uses Halton bases 2, 3, 5 for the three weight/leak dimensions.
    """
    h2 = _halton(HF, 2)
    h3 = _halton(HF, 3)
    h5 = _halton(HF, 5)

    w1    = np.round(h2 * 254 - 127).astype(np.int32)        # [-127, 127]
    w2    = np.round(h3 * 254 - 127).astype(np.int32)
    leaks = np.floor(h5 * (Config.MAX_LEAK_SHIFT + 1)).astype(np.int32)
    leaks = np.clip(leaks, 0, Config.MAX_LEAK_SHIFT)
    t1    = np.array([np.random.randint(Config.T1_MIN, Config.T1_MAX+1)], dtype=np.int32)
    t2    = np.array([np.random.randint(Config.T2_MIN, Config.T2_MAX+1)], dtype=np.int32)
    return np.concatenate([w1, w2, leaks, t1, t2])


def random_genome():
    w1    = np.random.randint(-Config.MAX_WEIGHT, Config.MAX_WEIGHT+1, HF)
    w2    = np.random.randint(-Config.MAX_WEIGHT, Config.MAX_WEIGHT+1, HF)
    leaks = np.random.randint(0, Config.MAX_LEAK_SHIFT+1,              HF)
    t1    = np.random.randint(Config.T1_MIN, Config.T1_MAX+1,          1)
    t2    = np.random.randint(Config.T2_MIN, Config.T2_MAX+1,          1)
    return np.concatenate([w1, w2, leaks, t1, t2]).astype(np.int32)


def decode(chrom):
    return (chrom[0:HF], chrom[HF:2*HF],
            chrom[2*HF:3*HF], int(chrom[3*HF]), int(chrom[-1]))


def clip_genome(chrom):
    c = chrom.copy()
    c[0:2*HF]    = np.clip(c[0:2*HF],   -Config.MAX_WEIGHT,   Config.MAX_WEIGHT)
    c[2*HF:3*HF] = np.clip(c[2*HF:3*HF], 0,                  Config.MAX_LEAK_SHIFT)
    c[3*HF]      = np.clip(c[3*HF],       Config.T1_MIN,       Config.T1_MAX)
    c[-1]        = np.clip(c[-1],          Config.T2_MIN,       Config.T2_MAX)
    return c


def tournament(pop, fitness, k=Config.TOURNAMENT_SIZE):
    idx = np.random.randint(0, len(pop), k)
    return pop[idx[np.argmax(fitness[idx])]]


def crossover(p1, p2):
    mask = np.random.rand(GENE_SIZE) < 0.5
    return np.where(mask, p1, p2).copy()


def mutate(chrom, rate, mut_w, mut_t1, mut_t2):
    c = chrom.copy()
    if np.random.rand() < rate:
        c[0:2*HF]    += np.random.randint(-mut_w,  mut_w+1,  2*HF)
        c[2*HF:3*HF] += np.random.randint(-1,       2,        HF)
        c[3*HF]      += np.random.randint(-mut_t1,  mut_t1+1)
        c[-1]        += np.random.randint(-mut_t2,  mut_t2+1)
    return clip_genome(c)

# ==============================================================================
# 7. PHASE 1 — ISLAND-MODEL GENETIC ALGORITHM
# ==============================================================================
def evaluate_island(island, X_t, Y_t, golden_mask_t, bonus_weight):
    """
    Evaluates one island (list of genomes) in sub-batches.
    Returns fitness array (ISLAND_POP,).
    """
    n_ind   = len(island)
    fitness = np.zeros(n_ind, dtype=np.float32)

    for start in range(0, n_ind, Config.SUB_BATCH):
        end     = min(start + Config.SUB_BATCH, n_ind)
        sub     = island[start:end]
        sub_n   = len(sub)

        W1_l, W2_l, L_l, T1_l, T2_l = [], [], [], [], []
        for ch in sub:
            w1, w2, lk, t1, t2 = decode(ch)
            W1_l.append(w1); W2_l.append(w2); L_l.append(lk)
            T1_l.append(t1); T2_l.append(t2)

        W1_t = torch.tensor(np.stack(W1_l), dtype=torch.int32, device=device)
        W2_t = torch.tensor(np.stack(W2_l), dtype=torch.int32, device=device)
        LK_t = torch.tensor(np.stack(L_l),  dtype=torch.int32, device=device)
        T1_t = torch.tensor(np.array(T1_l), dtype=torch.int32, device=device)
        T2_t = torch.tensor(np.array(T2_l), dtype=torch.int32, device=device)

        with torch.no_grad():
            X_b   = X_t.unsqueeze(0).expand(sub_n, -1, -1)
            preds = simulate_population_ga(X_b, W1_t, W2_t, LK_t, T1_t, T2_t)

            base  = balanced_accuracy(preds, Y_t)
            bonus = timbre_bonus(preds, golden_mask_t) if bonus_weight > 0 else \
                    torch.zeros(sub_n, device=device)

            fitness[start:end] = (base + bonus_weight * bonus).cpu().numpy()

    return fitness


def run_ga(X_train, Y_train):
    print(f"\n{'='*70}")
    print("🧬 PHASE 1 — ISLAND-MODEL GENETIC ALGORITHM")
    print(f"   {Config.N_ISLANDS} islands × {Config.ISLAND_POP} individuals = "
          f"{Config.N_ISLANDS * Config.ISLAND_POP} genomes")
    print(f"{'='*70}")

    # ── Pre-compute Golden Mask features ─────────────────────────────────────
    print("📊 Computing MFCC_11 / MFCC_12 Golden Mask features...")
    mfcc_feats  = extract_mfcc_features(X_train)
    golden_mask = build_golden_mask(mfcc_feats)
    drone_cover = np.mean(golden_mask[Y_train == 1]) * 100
    print(f"   Drone windows inside 2σ box: {drone_cover:.1f}%  "
          f"(should be ~95% for correct bounds)")

    X_t    = torch.tensor(X_train.astype(np.int32), dtype=torch.int32, device=device)
    Y_t    = torch.tensor(Y_train,                  dtype=torch.int32, device=device)
    gold_t = torch.tensor(golden_mask,              dtype=torch.bool,  device=device)

    # ── Initialise islands with quasi-random genomes ──────────────────────────
    islands = []
    for isl in range(Config.N_ISLANDS):
        island = []
        for j in range(Config.ISLAND_POP):
            # First half of each island: quasi-random; second half: pure random
            g = quasi_random_genome() if j < Config.ISLAND_POP // 2 else random_genome()
            island.append(g)
        islands.append(island)

    best_global_genome  = None
    best_global_fitness = -np.inf
    stag_counters       = [0] * Config.N_ISLANDS
    mut_rates           = [Config.MUTATION_RATE] * Config.N_ISLANDS
    t0 = time.time()

    for gen in range(Config.GENERATIONS):

        # Bonus weight ramp
        if gen < Config.RAW_ONLY_GENERATIONS:
            bw = 0.0
        else:
            ramp = (gen - Config.RAW_ONLY_GENERATIONS) / \
                   (Config.BONUS_RAMP_END - Config.RAW_ONLY_GENERATIONS)
            bw = Config.MAX_BONUS_WEIGHT * min(1.0, ramp)

        # Decaying mutation step sizes
        prog   = gen / Config.GENERATIONS
        mut_w  = max(5,   int(127  * (1.0 - prog)))
        mut_t1 = max(20,  int(1000 * (1.0 - prog)))
        mut_t2 = max(2,   int(20   * (1.0 - prog)))

        # ── Evolve each island independently ─────────────────────────────────
        island_bests = []
        for isl_idx, island in enumerate(islands):

            fitness = evaluate_island(island, X_t, Y_t, gold_t, bw)
            order   = np.argsort(fitness)[::-1]
            island  = [island[i] for i in order]
            fitness = fitness[order]

            if fitness[0] > best_global_fitness:
                best_global_fitness = fitness[0]
                best_global_genome  = island[0].copy()
                stag_counters[isl_idx] = 0
                mut_rates[isl_idx] = max(0.02, mut_rates[isl_idx] * 0.97)
                w1, w2, lk, t1, t2 = decode(best_global_genome)
                print(f"Gen {gen+1:03d} Isl {isl_idx} | "
                      f"Fit: {best_global_fitness:.4f} | "
                      f"T1={t1} T2={t2} | BonusW={bw:.3f} | "
                      f"{int(time.time()-t0)}s")
            else:
                stag_counters[isl_idx] += 1

            # Stagnation shock
            if stag_counters[isl_idx] >= Config.STAGNATION_LIMIT:
                mut_rates[isl_idx] = Config.STAGNATION_BOOST
                stag_counters[isl_idx] = 0
                print(f"   ⚡ Isl {isl_idx} stagnation shock @ gen {gen+1}")

            # Reproduce
            elite = Config.ELITE_PER_ISLAND
            next_isl = island[:elite]

            mr = mut_rates[isl_idx]
            while len(next_isl) < int(Config.ISLAND_POP * 0.80):
                p1 = tournament(island, fitness)
                p2 = tournament(island, fitness)
                if np.random.rand() < Config.CROSSOVER_RATE:
                    child = crossover(p1, p2)
                else:
                    child = p1.copy()
                next_isl.append(mutate(child, mr, mut_w, mut_t1, mut_t2))

            while len(next_isl) < Config.ISLAND_POP:
                next_isl.append(random_genome())

            islands[isl_idx] = next_isl
            island_bests.append(island[0].copy())

        # ── Migration every MIGRATION_EVERY generations ───────────────────────
        if (gen + 1) % Config.MIGRATION_EVERY == 0:
            print(f"   🔀 Gen {gen+1}: Ring migration between islands")
            for isl_idx in range(Config.N_ISLANDS):
                src  = island_bests[isl_idx]
                dest = (isl_idx + 1) % Config.N_ISLANDS
                # Replace worst individual in destination island
                islands[dest][-1] = src.copy()
                if Config.MIGRATION_COUNT > 1:
                    islands[dest][-2] = src.copy()

        # Cache flush once per generation
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ GA Complete | Best fitness: {best_global_fitness:.4f}")
    return best_global_genome

# ==============================================================================
# 8. PHASE 2 — SURROGATE GRADIENT SGD WITH STRAIGHT-THROUGH ESTIMATOR
# ==============================================================================

class QuantizeWeights(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for 8-bit weight quantization.
    Forward : clamp and round to integer in [-127, 127].
    Backward: gradient passes through unaltered (no quantization of gradient).
    """
    @staticmethod
    def forward(ctx, weights):
        return weights.clamp(-127, 127).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # STE — gradient flows straight through


class SurrogateSpike(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient for the Heavyside spiking function.
    Forward : Heavyside step (mem > threshold → 1, else 0).
    Backward: σ'(x) = σ(x)(1-σ(x)) evaluated at scaled membrane.
    """
    @staticmethod
    def forward(ctx, membrane, threshold):
        spike = (membrane > threshold).float()
        ctx.save_for_backward(membrane - threshold)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors
        sigma  = torch.sigmoid(diff)
        sg     = sigma * (1.0 - sigma)   # sigmoid derivative
        return grad_output * sg, None    # no gradient for threshold (frozen)


class MemboBrainSGD(nn.Module):
    """
    VHDL-accurate differentiable SNN for Phase 2 SGD.
    Discrete params (leaks, T1, T2) are frozen from GA champion.
    Only W1, W2 are trainable via STE + surrogate gradient.
    """
    def __init__(self, w1_init, w2_init, leaks, t1, t2):
        super().__init__()
        # Trainable — continuous float, quantized in forward via STE
        self.W1 = nn.Parameter(torch.tensor(w1_init.astype(np.float32)))
        self.W2 = nn.Parameter(torch.tensor(w2_init.astype(np.float32)))

        # Frozen discrete architecture from GA
        self.register_buffer('LEAKS', torch.tensor(leaks, dtype=torch.int32))
        self.T1 = float(t1)
        self.T2 = float(t2)

    def forward(self, X_batch):
        """
        X_batch : (B, WINDOW_SIZE)  signed float32 (decoded from int8)
        Returns  : (B,) total spike count per window (float for MSE loss)
        """
        B = X_batch.shape[0]

        # Quantize weights via STE
        W1_q = QuantizeWeights.apply(self.W1)   # (HF,)
        W2_q = QuantizeWeights.apply(self.W2)   # (HF,)

        mem1   = torch.zeros(B, HF, device=X_batch.device)
        mem2   = torch.zeros(B,     device=X_batch.device)
        spk2_total = torch.zeros(B, device=X_batch.device)

        leaks_f = self.LEAKS.float()   # for differentiable right-shift approximation

        for step in range(Config.WINDOW_SIZE):
            x_t = torch.abs(X_batch[:, step]).unsqueeze(1)  # AC Fix: (B, 1)

            cur1 = x_t * W1_q.unsqueeze(0)   # (B, HF)

            # Differentiable leak: approximate right-shift as division by 2^leak
            leak_factor = (2.0 ** leaks_f).unsqueeze(0)   # (1, HF)
            mem1_leaked = mem1 / leak_factor

            mem1_new = mem1_leaked + cur1
            spk1     = SurrogateSpike.apply(mem1_new, self.T1)      # (B, HF)
            mem1     = mem1_new * (1.0 - spk1.detach())             # reset on spike

            cur2 = torch.sum(spk1 * W2_q.unsqueeze(0), dim=1)      # (B,)
            mem2_leaked = mem2 / 2.0
            mem2_new    = mem2_leaked + cur2
            spk2        = SurrogateSpike.apply(mem2_new, self.T2)   # (B,)
            mem2        = mem2_new * (1.0 - spk2.detach())

            spk2_total += spk2

        return spk2_total   # (B,) total output spikes per window


def run_sgd(best_ga_genome, X_train, Y_train, X_val, Y_val):
    print(f"\n{'='*70}")
    print("🎓 PHASE 2 — SURROGATE GRADIENT SGD (STE + Sigmoid Surrogate)")
    print(f"{'='*70}")

    w1, w2, leaks, t1, t2 = decode(best_ga_genome)
    print(f"   Champion genome: T1={t1}, T2={t2}")
    print(f"   Leaks: {list(leaks)}")

    model = MemboBrainSGD(w1, w2, leaks, t1, t2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.SGD_LR)
    criterion = nn.MSELoss()

    # Convert to float32 for SGD (signed int8 decoded to AC waveform)
    X_tr_f = torch.tensor(X_train.astype(np.float32) / 127.0,
                           dtype=torch.float32, device=device)
    Y_tr   = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_vl_f = torch.tensor(X_val.astype(np.float32) / 127.0,
                           dtype=torch.float32, device=device)
    Y_vl   = torch.tensor(Y_val, dtype=torch.float32, device=device)

    # Target spike counts: drone → 12 spikes, noise → 0 spikes
    T_tr = Y_tr * Config.TARGET_SPIKES_DRONE
    T_vl = Y_vl * Config.TARGET_SPIKES_DRONE

    best_val_loss = np.inf
    patience_ctr  = 0
    best_weights  = (w1.copy(), w2.copy())

    n_train = len(X_tr_f)
    n_val   = len(X_vl_f)

    for epoch in range(Config.SGD_EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        perm     = torch.randperm(n_train, device=device)
        tr_loss  = 0.0
        n_batches = 0

        for start in range(0, n_train, Config.SGD_BATCH_SIZE):
            idx   = perm[start : start + Config.SGD_BATCH_SIZE]
            X_b   = X_tr_f[idx]
            T_b   = T_tr[idx]

            optimizer.zero_grad()
            spikes = model(X_b)
            loss   = criterion(spikes, T_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            n_batches += 1

        tr_loss /= n_batches

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_spikes = model(X_vl_f)
            val_loss   = criterion(val_spikes, T_vl).item()

        # Track improvement
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ctr  = 0
            # Extract quantized weights
            W1_q = model.W1.clamp(-127, 127).round().detach().cpu().numpy().astype(np.int32)
            W2_q = model.W2.clamp(-127, 127).round().detach().cpu().numpy().astype(np.int32)
            best_weights = (W1_q.copy(), W2_q.copy())
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or patience_ctr == 0:
            print(f"Epoch {epoch+1:03d} | TR Loss: {tr_loss:.4f} | "
                  f"VAL Loss: {val_loss:.4f} | Patience: {patience_ctr}/{Config.EARLY_STOP_PAT}")

        if patience_ctr >= Config.EARLY_STOP_PAT:
            print(f"\n⏹  Early stopping at epoch {epoch+1} "
                  f"(val_loss={best_val_loss:.4f})")
            break

    print(f"✅ SGD Complete | Best val loss: {best_val_loss:.4f}")
    return best_weights[0], best_weights[1]

# ==============================================================================
# 9. FINAL EVALUATION
# ==============================================================================
def final_eval(genome, X_test, Y_test):
    print(f"\n{'='*70}")
    print("🏆 FINAL EVALUATION — PURE EXTERNAL TEST SET")
    print(f"{'='*70}")

    w1, w2, leaks, t1, t2 = decode(genome)

    W1_t = torch.tensor(w1,    dtype=torch.int32, device=device).view(1, -1)
    W2_t = torch.tensor(w2,    dtype=torch.int32, device=device).view(1, -1)
    LK_t = torch.tensor(leaks, dtype=torch.int32, device=device).view(1, -1)
    T1_t = torch.tensor([t1],  dtype=torch.int32, device=device).view(1, 1)
    T2_t = torch.tensor([t2],  dtype=torch.int32, device=device)

    X_t = torch.tensor(X_test.astype(np.int32), dtype=torch.int32, device=device)
    with torch.no_grad():
        preds = simulate_population_ga(
            X_t.unsqueeze(0), W1_t, W2_t, LK_t, T1_t, T2_t
        )[0].cpu().numpy()

    MACRO = 15
    print(f"\nMacro-Window ROC (window = {MACRO}):")
    print(f"{'Threshold':<12} {'Recall':>10} {'FAR':>10} {'Accuracy':>12}")
    print("-" * 46)
    for thresh in [1, 3, 5, 8, 10, 12, 15]:
        MTP = MTN = MFP = MFN = 0
        for i in range(0, len(preds) - MACRO + 1, MACRO):
            mp = preds[i:i+MACRO]; mt = Y_test[i:i+MACRO]
            alarm  = 1 if np.sum(mp) >= thresh else 0
            actual = 1 if np.sum(mt) > (MACRO // 2) else 0
            if   alarm and actual:          MTP += 1
            elif not alarm and not actual:  MTN += 1
            elif alarm and not actual:      MFP += 1
            else:                           MFN += 1
        rec  = MTP / (MTP + MFN + 1e-6) * 100
        far  = MFP / (MFP + MTN + 1e-6) * 100
        macc = (MTP + MTN) / (MTP + MTN + MFP + MFN + 1e-6) * 100
        print(f"{thresh:<12} {rec:>9.1f}% {far:>9.1f}% {macc:>11.2f}%")

    print(f"\n🏆 Final Genome:")
    print(f"   W1    = {list(w1)}")
    print(f"   W2    = {list(w2)}")
    print(f"   LEAKS = {list(leaks)}")
    print(f"   T1    = {t1}  |  T2 = {t2}")

# ==============================================================================
# 10. GENOME ASSEMBLY & SAVE
# ==============================================================================
def assemble_and_save(ga_genome, sgd_w1, sgd_w2):
    """
    Replace GA weights with SGD-refined weights, keep discrete params from GA.
    Save as best_memetic_genome.npy in canonical genome layout.
    """
    _, _, leaks, t1, t2 = decode(ga_genome)
    final = np.concatenate([
        sgd_w1.astype(np.int32),
        sgd_w2.astype(np.int32),
        leaks.astype(np.int32),
        np.array([t1], dtype=np.int32),
        np.array([t2], dtype=np.int32)
    ])
    np.save(Config.GENOME_SAVE, final)
    print(f"\n💾 Final memetic genome saved to {Config.GENOME_SAVE}")
    return final

# ==============================================================================
# 11. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    t_start = time.time()
    print("="*70)
    print("🚀 MEMETIC HYBRID SNN — MEMBO DRONE DETECTOR")
    print("   Phase 1: Island GA + Fisher Timbre Tutor (MFCC_11/MFCC_12)")
    print("   Phase 2: Surrogate Gradient SGD with STE")
    print("="*70)

    # Build datasets
    X_train, Y_train, X_val, Y_val, X_test, Y_test = build_datasets()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    ga_champion = run_ga(X_train, Y_train)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    refined_w1, refined_w2 = run_sgd(ga_champion, X_train, Y_train, X_val, Y_val)

    # ── Assemble final genome ─────────────────────────────────────────────────
    final_genome = assemble_and_save(ga_champion, refined_w1, refined_w2)

    # ── Evaluate on pure external test set ───────────────────────────────────
    final_eval(final_genome, X_test, Y_test)

    print(f"\n🕒 Total runtime: {int(time.time() - t_start)} seconds")
