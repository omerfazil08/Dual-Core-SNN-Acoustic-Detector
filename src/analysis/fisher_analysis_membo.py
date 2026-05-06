import json
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 🔧 Updated to point to the membo_phase folder
JSON_PATH = 'C:/Users/omer-/Desktop/drone_audio_detector/drone_analysis_data/membo_phase/master_analysis/master_summary.json'

def analyze_fisher_ratios():
    print(f"🔍 Loading master summary from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        master_summary = json.load(f)

    # Look at the 'full_train' dataset
    full_stats = None
    for subset in master_summary['subsets']:
        if subset['subset_name'] == 'full_train':
            full_stats = subset
            break
    
    if full_stats is None:
        full_stats = master_summary['subsets'][-1]

    feature_names = master_summary['feature_names']
    drone_mean = np.array(full_stats['drone_mean'])
    drone_std = np.array(full_stats['drone_std'])
    noise_mean = np.array(full_stats['noise_mean'])
    noise_std = np.array(full_stats['noise_std'])

    # ==========================================================================
    # 🧮 FISHER DISCRIMINANT RATIO FORMULA
    # ==========================================================================
    fisher_ratios = (drone_mean - noise_mean)**2 / (drone_std**2 + noise_std**2)

    results = [(feature_names[i], fisher_ratios[i]) for i in range(len(feature_names))]
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    print("🏆 TOP 10 FEATURES FOR MAMBO DETECTION (FISHER RATIO)")
    print("="*50)
    for name, score in results[:10]:
        print(f" {name:<12} : {score:.4f}")

    print("\n" + "="*50)
    print("🎯 TIMBRE TUTOR 2-SIGMA BOUNDING BOX PARAMETERS")
    print("="*50)
    
    # Dynamically extract bounds for the absolute top 2 features
    top_2_features = [results[0][0], results[1][0]]
    
    for target in top_2_features:
        idx = feature_names.index(target)
        d_m = drone_mean[idx]
        d_s = drone_std[idx]
        
        # Calculate 2-Sigma bounds (95% statistical coverage)
        lower_bound = d_m - (2 * d_s)
        upper_bound = d_m + (2 * d_s)
        
        print(f"--- {target} (Index: {idx}) ---")
        print(f"  Drone : Mean = {d_m:8.4f} | Std = {d_s:8.4f}")
        print(f"  > Recommended 2-Sigma Bounding Box: [{lower_bound:.1f} to {upper_bound:.1f}]\n")

if __name__ == "__main__":
    analyze_fisher_ratios()