from pathlib import Path

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "IRIS and FINGERPRINT DATASET"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "fingerprint"
)

PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
