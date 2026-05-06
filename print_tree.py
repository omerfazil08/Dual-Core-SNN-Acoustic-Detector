import os

def generate_tree(dir_path, prefix=""):
    # Folders we want to ignore so they don't flood the output
    IGNORE_DIRS = {'.git', '__pycache__', 'data', 'vivado_project', '.ipynb_checkpoints'}
    
    try:
        # Get all items, sort directories first, then files
        items = os.listdir(dir_path)
        items.sort(key=lambda x: (not os.path.isdir(os.path.join(dir_path, x)), x.lower()))
    except PermissionError:
        return

    # Filter out ignored directories
    items = [item for item in items if item not in IGNORE_DIRS]
    
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        # Print the current item
        print(f"{prefix}{connector}{item}")
        
        # If it's a directory, recursively call the function
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            generate_tree(path, prefix + extension)

if __name__ == "__main__":
    print("Dual-Core-SNN-Acoustic-Detector/")
    # Run from the current directory
    generate_tree(".")
    print("\nNote: The 'data' folder and cache directories are intentionally hidden to keep the printout clean.")