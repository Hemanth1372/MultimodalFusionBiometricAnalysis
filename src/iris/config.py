from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

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
    / "quality_check"
)

DEBUG_DIR = OUTPUT_DIR / "debug"

NORMALIZED_DIR = OUTPUT_DIR / "normalized"

REPORT_PATH = OUTPUT_DIR / "quality_report.csv"


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SAVE_DEBUG_IMAGES = True

SAVE_NORMALIZED_IMAGES = True


# ============================================================
# NORMALIZATION
# ============================================================
NORMALIZED_HEIGHT = 128
NORMALIZED_WIDTH = 256

RADIAL_RESOLUTION = 128

ANGULAR_RESOLUTION = 256


# ============================================================
# QUALITY ANALYSIS
# ============================================================

BOUNDARY_SAMPLES = 72


# ============================================================
# PUPIL DETECTION
# ============================================================

PUPIL_MIN_RADIUS = 20

PUPIL_MAX_RADIUS = 35


# ============================================================
# IRIS DETECTION
# ============================================================

IRIS_MIN_RADIUS = 60

IRIS_MAX_RADIUS = 72

PUPIL_IRIS_CENTER_RANGE = 4


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NORMALIZED_DIR.mkdir(
    parents=True,
    exist_ok=True
)