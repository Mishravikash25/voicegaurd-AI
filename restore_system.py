"""
VoiceGuard AI - Instant Restoration Utility
Use this script if you ever want to roll back the model or profile state
to this stable checkpoint.
"""
import shutil
from pathlib import Path

def restore():
    backup_dir = Path("RESTORE_POINT_BACKUP")
    if not backup_dir.exists():
        print("Error: RESTORE_POINT_BACKUP directory not found!")
        return

    # 1. Restore Model Artifacts
    models_dst = Path("backend/ml_system/models")
    models_dst.mkdir(parents=True, exist_ok=True)
    for pkl in (backup_dir / "models").glob("*.pkl"):
        shutil.copy(pkl, models_dst / pkl.name)
        print(f"Restored model: {pkl.name}")

    # 2. Restore Enrolled Profiles
    profiles_dst = Path("backend/enrolled_profiles")
    profiles_dst.mkdir(parents=True, exist_ok=True)
    for f in (backup_dir / "enrolled_profiles").glob("*"):
        if f.is_file():
            shutil.copy(f, profiles_dst / f.name)
            print(f"Restored profile file: {f.name}")

    print("\n[SUCCESS] RESTORATION COMPLETE! System successfully rolled back to stable state.")

if __name__ == "__main__":
    restore()
