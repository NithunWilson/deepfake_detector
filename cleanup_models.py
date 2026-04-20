import os
import shutil

# Keep only the compatible models
models_to_keep = [
    "deepfake_model_84_60.pth",
    "best_model_85_60.pth", 
    "best_model_90_60.pth",
    "model_metadata_84.json"
]

# Directory to move old models
backup_dir = "models_backup"
os.makedirs(backup_dir, exist_ok=True)

# Move incompatible models to backup
for filename in os.listdir("models"):
    if filename not in models_to_keep and filename != "face_detection":
        old_path = os.path.join("models", filename)
        new_path = os.path.join(backup_dir, filename)
        shutil.move(old_path, new_path)
        print(f"Moved: {filename}")

print("\nCleanup complete!")
print("Kept models:", models_to_keep)
print("Old models moved to:", backup_dir)
