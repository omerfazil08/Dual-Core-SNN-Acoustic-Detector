import json
import numpy as np

# ==============================================================================
# CONFIGURATION: Point this to your actual JSON file location
# ==============================================================================
JSON_PATH = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data/master_analysis/master_summary.json'

def analyze_fisher_ratios():
    print(f"📂 Loading master summary from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        master_summary = json.load(f)

    # We want to look at the 'full_train' dataset because it has the most variance
    full_stats = None
    for subset in master_summary['subsets']:
        if subset['subset_name'] == 'full_train':
            full_stats = subset
            break
    
    # Fallback to the last one if full_train isn't explicitly named
    if full_stats is None:
        full_stats = master_summary['subsets'][-1]

    feature_names = master_summary['feature_names']
    drone_mean = np.array(full_stats['drone_mean'])
    drone_std = np.array(full_stats['drone_std'])
    noise_mean = np.array(full_stats['noise_mean'])
    noise_std = np.array(full_stats['noise_std'])

    # ==========================================================================
    # 🧮 FISHER DISCRIMINANT RATIO FORMULA
    # Calculates the distance between the Drone and Noise means, 
    # penalized by how much variance (standard deviation) they have.
    # ==========================================================================
    fisher_ratios = (drone_mean - noise_mean)**2 / (drone_std**2 + noise_std**2)

    # Pair names with their scores and sort highest to lowest
    results = [(feature_names[i], fisher_ratios[i]) for i in range(len(feature_names))]
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    print("🏆 TOP 10 FEATURES FOR DRONE DETECTION (FISHER RATIO)")
    print("="*50)
    for name, score in results[:10]:
        print(f" {name:<12} : {score:.4f}")

    print("\n" + "="*50)
    print("🔍 TIMBRE TUTOR BOUNDING BOX PARAMETERS")
    print("="*50)
    
    # Print the exact parameters for the top 2 features to build the Tutor
    for target in ['Rolloff', 'MFCC_4']:
        if target in feature_names:
            idx = feature_names.index(target)
            d_m = drone_mean[idx]
            d_s = drone_std[idx]
            n_m = noise_mean[idx]
            n_s = noise_std[idx]
            
            print(f"--- {target} ---")
            print(f"  Drone : Mean = {d_m:8.4f} | Std = {d_s:8.4f}")
            print(f"  Noise : Mean = {n_m:8.4f} | Std = {n_s:8.4f}")
            
            # Recommend a bounding box (roughly Mean +/- 1 Standard Deviation)
            print(f"  > Recommended SNN Bounding Box: [{d_m - d_s:.1f} to {d_m + d_s:.1f}]\n")

if __name__ == "__main__":
    analyze_fisher_ratios()