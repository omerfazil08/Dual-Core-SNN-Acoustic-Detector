# src/utils/build_hostile_bebop.py
import os
import shutil
import glob

def build_hostile_folder():
    base_data_dir = "../../data"
    noise_source = os.path.join(base_data_dir, "unknown")
    membo_source = os.path.join(base_data_dir, "membo_1")
    hostile_dest = os.path.join(base_data_dir, "hostile_noise_for_bebop")

    print("🚀 Building Hostile Negative Mining Sandbox for Bebop...")
    
    # Create the folder
    os.makedirs(hostile_dest, exist_ok=True)
    
    # Copy 10,000 noise files
    print(f"Copying background noise from {noise_source}...")
    for f in glob.glob(os.path.join(noise_source, "*.wav")):
        shutil.copy(f, hostile_dest)
        
    # INJECT MEMBO (The Hostile Pollutant)
    print(f"Injecting Membo acoustic pollutant from {membo_source}...")
    for f in glob.glob(os.path.join(membo_source, "*.wav")):
        shutil.copy(f, hostile_dest)
        
    print(f"✅ Hostile folder built successfully at: {hostile_dest}")

if __name__ == "__main__":
    build_hostile_folder()