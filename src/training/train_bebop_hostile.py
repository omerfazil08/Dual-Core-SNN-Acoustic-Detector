# ==============================================================================
# CLEAN bebop SNN TRAINING ENGINE (Raw Audio / No Earplugs)
# Target: MFCC 11 & 12 (High Pitch Whine) vs Standard Noise
# ==============================================================================

import os, json, glob, time, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from scipy.io import wavfile
from scipy import signal as scipy_signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Clean bebop Training Engine on: {device}")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    TARGET_DRONE = "bebop" 
    
    DRONE_FOLDER = "../../data/bebop_1" # Use relative paths!
    
    # ⚠️ THIS MUST BE THE HOSTILE FOLDER (Noise + Membo Audio)
    NOISE_FOLDER = "../../data/hostile_noise_for_bebop" 
    
    # Bebop Fisher Frequencies (Deep Rotor Hum)
    MFCC_A, MFCC_B = 0, 3  

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 248
    HIDDEN_SIZE = 64

    MAX_WEIGHT = 127
    MAX_LEAK_SHIFT = 3
    T1_MIN, T1_MAX = 500, 2000 
    T2_MIN, T2_MAX = 2, 40

    NUM_ISLANDS = 10
    INDIVIDUALS_PER_ISLAND = 30
    GENERATIONS = 150
    MUTATION_RATE = 0.05
    ELITE_RATIO = 0.10

    RAW_ONLY_GENERATIONS = 20
    BONUS_RAMP_START = 20
    BONUS_RAMP_END = 80
    MAX_BONUS_WEIGHT = 0.3

    FILES_PER_CLASS = 200 
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10

    FINE_TUNE_EPOCHS = 150
    LEARNING_RATE = 0.1
    EARLY_STOP_PATIENCE = 150

    FAR_PENALTY = 3.0          
    TARGET_SPIKES_DRONE = 12.0 
    TARGET_SPIKES_NOISE = 0.0  

# ==============================================================================
# PHASE 0: FISHER FEATURES (Dynamic Timbre Tutor)
# ==============================================================================
def extract_fisher_features(windows_8bit):
    N = len(windows_8bit)
    features = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        sig = windows_8bit[i].astype(np.float32) / 127.0
        rolloff = librosa.feature.spectral_rolloff(
            y=sig, sr=Config.SAMPLE_RATE,
            n_fft=Config.WINDOW_SIZE, hop_length=Config.WINDOW_SIZE)
        centroid = librosa.feature.spectral_centroid(
            y=sig, sr=Config.SAMPLE_RATE,
            n_fft=Config.WINDOW_SIZE, hop_length=Config.WINDOW_SIZE)
        features[i, 0] = rolloff[0, 0]
        features[i, 1] = centroid[0, 0]
        mfcc = librosa.feature.mfcc(
            y=sig, sr=Config.SAMPLE_RATE, n_mfcc=13,
            n_fft=Config.WINDOW_SIZE, hop_length=Config.WINDOW_SIZE)
        features[i, 2] = mfcc[Config.MFCC_A, 0] 
        features[i, 3] = mfcc[Config.MFCC_B, 0] 
    return features

def compute_fisher_stats(X_train, Y_train):
    print(f"📐 Computing Fisher stats targeting bebop frequencies...")
    feats = extract_fisher_features(X_train)
    drone = feats[Y_train == 1]
    noise = feats[Y_train == 0]
    return drone.mean(axis=0), drone.std(axis=0), noise.mean(axis=0), noise.std(axis=0)

def compute_window_scores(fisher_features, d_mean, d_std, n_mean, n_std):
    N = fisher_features.shape[0]
    scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ratio_product = 1.0
        for j in range(4):
            x = fisher_features[i, j]
            drone_pdf = (1.0 / (math.sqrt(2*math.pi) * d_std[j])) * math.exp(-0.5 * ((x - d_mean[j]) / d_std[j])**2)
            noise_pdf = (1.0 / (math.sqrt(2*math.pi) * n_std[j])) * math.exp(-0.5 * ((x - n_mean[j]) / n_std[j])**2)
            ratio = (drone_pdf + 1e-12) / (drone_pdf + noise_pdf + 1e-12)
            ratio_product *= ratio
        scores[i] = np.clip(ratio_product, 0.0, 1.0)
    return scores

# ==============================================================================
# SNN CORE & PYTORCH MODULES
# ==============================================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem, threshold)
        return (mem > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        mem, thr = ctx.saved_tensors
        alpha = 2.0
        grad = grad_output * (alpha / (2 * torch.cosh(alpha * (mem - thr)))**2)
        return grad, None

class STEQuantizeInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        return torch.clamp(torch.round(weights), -127, 127)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

spike_fn = SurrogateSpike.apply
quantize_fn = STEQuantizeInt8.apply

class FineTuneModel(nn.Module):
    def __init__(self, w1, w2, leaks, t1, t2):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(w1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(w2, dtype=torch.float32))
        self.register_buffer('leaks', torch.tensor(leaks, dtype=torch.float32))
        self.register_buffer('t1', torch.tensor([t1], dtype=torch.float32))
        self.register_buffer('t2', torch.tensor([t2], dtype=torch.float32))

    def forward(self, x):
        batch, win = x.shape
        H = self.w1.shape[0]
        w1_q = quantize_fn(self.w1).view(1, H)
        w2_q = quantize_fn(self.w2).view(1, H)
        mem1 = torch.zeros((batch, H), dtype=torch.float32, device=x.device)
        mem2 = torch.zeros(batch, dtype=torch.float32, device=x.device)
        total_spikes = torch.zeros(batch, dtype=torch.float32, device=x.device)
        leak_divisors = 2.0 ** self.leaks
        for step in range(win):
            xt = torch.abs(x[:, step]).unsqueeze(1)
            cur1 = xt * w1_q
            mem1 = torch.floor(mem1 / leak_divisors) + cur1
            spk1 = spike_fn(mem1, self.t1)
            mem1 = mem1 * (1.0 - spk1)
            cur2 = torch.sum(spk1 * w2_q, dim=1)
            mem2 = torch.floor(mem2 / 2.0) + cur2
            spk2 = spike_fn(mem2, self.t2)
            mem2 = mem2 * (1.0 - spk2)
            total_spikes += spk2
        return total_spikes

# ==============================================================================
# DATA LOADER (Raw Audio Only)
# ==============================================================================
def load_and_quantize(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1: data = np.mean(data, axis=1)
    if np.max(np.abs(data)) > 0: data = data.astype(np.float32) / np.max(np.abs(data))
    if sr != Config.SAMPLE_RATE:
        num_target = int(len(data) * Config.SAMPLE_RATE / sr)
        data = scipy_signal.resample(data, num_target)
    return np.clip(data * 127.0, -127, 127).astype(np.int32)

def build_datasets():
    np.random.seed(42) 
    drone_files = glob.glob(os.path.join(Config.DRONE_FOLDER, "*.wav"))
    noise_files = glob.glob(os.path.join(Config.NOISE_FOLDER, "*.wav"))
    np.random.shuffle(drone_files); np.random.shuffle(noise_files)
    
    drone_files = drone_files[:Config.FILES_PER_CLASS]
    noise_files = noise_files[:Config.FILES_PER_CLASS]
    split = int(Config.FILES_PER_CLASS * Config.TRAIN_RATIO)
    
    train_files = [(f,1) for f in drone_files[:split]] + [(f,0) for f in noise_files[:split]]
    test_files  = [(f,1) for f in drone_files[split:]] + [(f,0) for f in noise_files[split:]]
    
    def process(files):
        windows, labels = [], []
        for f, lbl in files:
            try:
                q = load_and_quantize(f)
                n_win = len(q) // Config.WINDOW_SIZE
                for i in range(n_win):
                    windows.append(q[i*Config.WINDOW_SIZE:(i+1)*Config.WINDOW_SIZE])
                    labels.append(lbl)
            except: pass
        return np.stack(windows), np.array(labels, dtype=np.int32)
        
    X_train, Y_train = process(train_files)
    X_test, Y_test = process(test_files)
    
    n_val = int(len(X_train) * Config.VAL_RATIO)
    idx = np.random.permutation(len(X_train))
    X_val, Y_val = X_train[idx[:n_val]], Y_train[idx[:n_val]]
    X_train, Y_train = X_train[idx[n_val:]], Y_train[idx[n_val:]]
    
    print(f"✅ Clean Build: {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# ==============================================================================
# PHASE 1: LAMARCKIAN GENETIC ALGORITHM
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

def decode_discrete(genome):
    H = Config.HIDDEN_SIZE
    return genome[:H].astype(np.int32), int(genome[H]), int(genome[H+1])

def run_island_ga(X_train, Y_train, d_mean, d_std, n_mean, n_std):
    print(f"\n🧬 PHASE 1: Lamarckian GA + Fisher Timbre Tutor")
    fisher_feats = extract_fisher_features(X_train)
    train_scores = compute_window_scores(fisher_feats, d_mean, d_std, n_mean, n_std)
    train_scores_t = torch.tensor(train_scores, dtype=torch.float32, device=device)

    X_tr_t = torch.tensor(X_train, dtype=torch.int32, device=device)
    Y_tr_t = torch.tensor(Y_train, dtype=torch.int32, device=device)
    
    X_nudge = torch.tensor(X_train[:512], dtype=torch.float32, device=device)
    Y_nudge = torch.tensor(Y_train[:512], dtype=torch.float32, device=device)
    T_nudge = torch.where(Y_nudge == 1, torch.tensor(Config.TARGET_SPIKES_DRONE, device=device), torch.tensor(Config.TARGET_SPIKES_NOISE, device=device))
    loss_fn = nn.MSELoss(reduction='none')

    def lamarckian_nudge(w1, w2, leaks, t1, t2):
        model = FineTuneModel(w1, w2, leaks, t1, t2).to(device)
        opt = optim.Adam(model.parameters(), lr=0.05) 
        for _ in range(3):
            opt.zero_grad()
            spikes = model(X_nudge)
            raw_loss = loss_fn(spikes, T_nudge)
            weights = torch.where(Y_nudge == 0, torch.tensor(Config.FAR_PENALTY, device=device), torch.tensor(1.0, device=device))
            loss = (raw_loss * weights).mean()
            loss.backward()
            opt.step()
        w1_n = torch.clamp(torch.round(model.w1.detach()), -127, 127).cpu().numpy().astype(np.int32)
        w2_n = torch.clamp(torch.round(model.w2.detach()), -127, 127).cpu().numpy().astype(np.int32)
        return w1_n, w2_n

    champions = []
    for island in range(Config.NUM_ISLANDS):
        print(f"\n🏝️  Island {island+1}/{Config.NUM_ISLANDS}")
        pop, weights = [], []
        for i in range(Config.INDIVIDUALS_PER_ISLAND):
            leaks = np.random.randint(0, Config.MAX_LEAK_SHIFT+1, Config.HIDDEN_SIZE)
            t1 = np.random.randint(Config.T1_MIN, Config.T1_MAX+1, 1)
            t2 = np.random.randint(Config.T2_MIN, Config.T2_MAX+1, 1)
            pop.append(np.concatenate([leaks, t1, t2]).astype(np.int32))
            weights.append((np.random.randint(-127, 128, Config.HIDDEN_SIZE).astype(np.int32), 
                            np.random.randint(-127, 128, Config.HIDDEN_SIZE).astype(np.int32)))

        best_fitness, best_genome, best_weights = -float('inf'), None, None

        for gen in range(Config.GENERATIONS):
            elite_count = max(1, int(len(pop) * Config.ELITE_RATIO))
            if gen > 0: 
                for i in range(elite_count):
                    leaks, t1, t2 = decode_discrete(pop[i])
                    w1_new, w2_new = lamarckian_nudge(weights[i][0], weights[i][1], leaks, t1, t2)
                    weights[i] = (w1_new, w2_new)

            W1 = torch.tensor(np.stack([w[0] for w in weights]), dtype=torch.int32, device=device)
            W2 = torch.tensor(np.stack([w[1] for w in weights]), dtype=torch.int32, device=device)
            LEAKS = torch.tensor(np.stack([g[:Config.HIDDEN_SIZE] for g in pop]), dtype=torch.int32, device=device)
            T1 = torch.tensor([g[Config.HIDDEN_SIZE] for g in pop], dtype=torch.int32, device=device)
            T2 = torch.tensor([g[Config.HIDDEN_SIZE+1] for g in pop], dtype=torch.int32, device=device)

            with torch.no_grad():
                X_batch = X_tr_t.unsqueeze(0).expand(len(pop), -1, -1)
                preds = simulate_population(X_batch, W1, W2, LEAKS, T1, T2)
                Y_exp = Y_tr_t.unsqueeze(0)
                TP = torch.sum((preds==1) & (Y_exp==1), dim=1).float()
                TN = torch.sum((preds==0) & (Y_exp==0), dim=1).float()
                FP = torch.sum((preds==1) & (Y_exp==0), dim=1).float()
                FN = torch.sum((preds==0) & (Y_exp==1), dim=1).float()
                TPR = TP / (TP + FN + 1e-6)
                FAR = FP / (FP + TN + 1e-6)
                
                base = TPR - (FAR * Config.FAR_PENALTY)
                bonus_w = 0.0
                if gen >= Config.RAW_ONLY_GENERATIONS:
                    ramp = min(1.0, (gen - Config.RAW_ONLY_GENERATIONS) / (Config.BONUS_RAMP_END - Config.RAW_ONLY_GENERATIONS))
                    bonus_w = Config.MAX_BONUS_WEIGHT * ramp
                    total_spikes = preds.sum(dim=1).float()
                    score_sum = (preds * train_scores_t.unsqueeze(0)).sum(dim=1).float()
                    bonus = score_sum / (total_spikes + 1e-6)
                    fit = base + bonus_w * bonus
                else:
                    fit = base
                    
                fitness = fit.cpu().numpy()

            order = np.argsort(fitness)[::-1]
            pop, weights, fitness = [pop[i] for i in order], [weights[i] for i in order], fitness[order]

            if fitness[0] > best_fitness:
                best_fitness = fitness[0]
                best_genome = pop[0].copy()
                best_weights = (weights[0][0].copy(), weights[0][1].copy())

            if gen % 15 == 0: print(f"   Gen {gen:03d} | Best fit: {best_fitness:.4f}")

            next_pop, next_weights = pop[:elite_count], weights[:elite_count]
            while len(next_pop) < len(pop):
                idx = np.random.randint(0, len(pop), 2)
                p1, w1_p = pop[idx[0]], weights[idx[0]]
                child = p1.copy()
                child[:Config.HIDDEN_SIZE] += np.random.randint(-1, 2, Config.HIDDEN_SIZE)
                child[Config.HIDDEN_SIZE] += np.random.randint(-100, 101)   
                child[Config.HIDDEN_SIZE+1] += np.random.randint(-3, 4)     
                child = np.clip(child, 0, [Config.MAX_LEAK_SHIFT]*Config.HIDDEN_SIZE + [Config.T1_MAX, Config.T2_MAX])
                next_pop.append(child)
                next_weights.append(w1_p)
            pop, weights = next_pop, next_weights

        print(f"✅ Island {island+1} champion fitness: {best_fitness:.4f}")
        champions.append({'genome': best_genome, 'w1': best_weights[0], 'w2': best_weights[1]})
    return champions

# ==============================================================================
# PHASE 2: FINE‑TUNE
# ==============================================================================
def fine_tune_champion(champ, X_train, Y_train, X_val, Y_val):
    w1 = champ['w1']; w2 = champ['w2']
    leaks, t1, t2 = decode_discrete(champ['genome'])
    model = FineTuneModel(w1, w2, leaks, t1, t2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_v = torch.tensor(Y_val, dtype=torch.float32, device=device)

    T_t = torch.where(Y_t == 1, torch.tensor(Config.TARGET_SPIKES_DRONE, device=device), torch.tensor(Config.TARGET_SPIKES_NOISE, device=device))
    T_v = torch.where(Y_v == 1, torch.tensor(Config.TARGET_SPIKES_DRONE, device=device), torch.tensor(Config.TARGET_SPIKES_NOISE, device=device))
    loss_fn = nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    patience = 0
    best_w1, best_w2 = w1, w2

    for epoch in range(Config.FINE_TUNE_EPOCHS):
        model.train()
        optimizer.zero_grad()
        spikes = model(X_t)
        weights = torch.where(Y_t == 0, torch.tensor(Config.FAR_PENALTY, device=device), torch.tensor(1.0, device=device))
        loss = (loss_fn(spikes, T_t) * weights).mean()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_spikes = model(X_v)
            v_weights = torch.where(Y_v == 0, torch.tensor(Config.FAR_PENALTY, device=device), torch.tensor(1.0, device=device))
            val_loss = (loss_fn(v_spikes, T_v) * v_weights).mean().item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience = 0
            best_w1 = torch.clamp(torch.round(model.w1.detach()), -127, 127).cpu().numpy().astype(np.int32)
            best_w2 = torch.clamp(torch.round(model.w2.detach()), -127, 127).cpu().numpy().astype(np.int32)
        else:
            patience += 1
            if patience >= Config.EARLY_STOP_PATIENCE:
                print(f"   ⏹ Early Stop Epoch {epoch} | Val Loss: {best_val_loss:.4f}")
                break

    return best_w1, best_w2

# ==============================================================================
# MAIN 
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"🚀 CLEAN TRAINING ENGINE: TARGETING bebop")
    print("=" * 80)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = build_datasets()
    d_mean, d_std, n_mean, n_std = compute_fisher_stats(X_train, Y_train)
    champions = run_island_ga(X_train, Y_train, d_mean, d_std, n_mean, n_std)

    print("\n🔧 PHASE 2: Fine‑tuning island champions...")
    best_overall = None
    best_macro_far = 100.0
    best_macro_rec = 0.0

    for idx, champ in enumerate(champions):
        print(f"\n🏝️  Fine‑tuning champion {idx+1}/{len(champions)}")
        w1_new, w2_new = fine_tune_champion(champ, X_train, Y_train, X_val, Y_val)
        champ['w1'] = w1_new
        champ['w2'] = w2_new

        leaks, t1, t2 = decode_discrete(champ['genome'])
        W1_t = torch.tensor(w1_new, dtype=torch.int32, device=device).view(1, -1)
        W2_t = torch.tensor(w2_new, dtype=torch.int32, device=device).view(1, -1)
        L_t  = torch.tensor(leaks, dtype=torch.int32, device=device).view(1, -1)
        T1_t = torch.tensor([t1], dtype=torch.int32, device=device)
        T2_t = torch.tensor([t2], dtype=torch.int32, device=device)

        X_tt = torch.tensor(X_test, dtype=torch.int32, device=device).unsqueeze(0)
        with torch.no_grad():
            micro_preds = simulate_population(X_tt, W1_t, W2_t, L_t, T1_t, T2_t)[0].cpu().numpy()
            
        MACRO = 64
        n_mac = len(micro_preds) // MACRO
        if n_mac < 5: continue
        
        m_sums = np.array([micro_preds[i*MACRO:(i+1)*MACRO].sum() for i in range(n_mac)])
        m_labels = np.array([1 if Y_test[i*MACRO:(i+1)*MACRO].mean() > 0.5 else 0 for i in range(n_mac)])
        
        champ_far = 100.0
        champ_rec = 0.0
        for T in range(1, MACRO):
            TP = TN = FP = FN = 0
            for i in range(n_mac - 2):
                alarm = 1 if (m_sums[i:i+3] >= T).sum() >= 2 else 0
                if alarm and m_labels[i]: TP += 1
                elif not alarm and not m_labels[i]: TN += 1
                elif alarm and not m_labels[i]: FP += 1
                else: FN += 1
            rec = TP/(TP+FN+1e-6)*100
            far = FP/(FP+TN+1e-6)*100
            
            if rec >= 80.0 and far < champ_far:
                champ_far = far
                champ_rec = rec
                
        print(f"   Test Macro Result -> Recall: {champ_rec:.1f}% | FAR: {champ_far:.1f}%")
        
        if champ_far < best_macro_far or (champ_far == best_macro_far and champ_rec > best_macro_rec):
            best_macro_far = champ_far
            best_macro_rec = champ_rec
            best_overall = champ

    if best_overall is not None:
        # Saving directly to Drive to prevent data loss
        save_path = "/content/drive/MyDrive/"
        np.save(save_path + f"best_{Config.TARGET_DRONE}_genome_finetuned.npy", best_overall['genome'])
        np.save(save_path + f"best_{Config.TARGET_DRONE}_weights_w1.npy", best_overall['w1'])
        np.save(save_path + f"best_{Config.TARGET_DRONE}_weights_w2.npy", best_overall['w2'])
        print(f"\n🏁 Best RAW bebop champion safely saved to Google Drive! Deployment Spec -> Recall: {best_macro_rec:.1f}% | FAR: {best_macro_far:.1f}%")