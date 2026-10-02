from pathlib import Path
from PIL import Image

DATASET = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
    r"\dataset\IRIS and FINGERPRINT DATASET"
)
subjects = [
    x for x in DATASET.iterdir()
    if x.is_dir() and x.name.isdigit()
]

for subject in sorted(subjects, key=lambda x: int(x.name)):
    print(f"\nSubject {subject.name}")

    for folder in ["Fingerprint", "left", "right"]:
        folder_path = subject / folder
        images = list(folder_path.glob("*.BMP"))

        print(f"  {folder}: {len(images)} images")

        if images:
            img = Image.open(images[0])
            print(f"    Example: {images[0].name}")
            print(f"    Size: {img.size}")
            print(f"    Mode: {img.mode}")