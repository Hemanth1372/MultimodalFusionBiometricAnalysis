import csv
import sys
from pathlib import Path

import cv2


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Hemanth\OneDrive\Desktop\Multimodel_Biometrics"
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.iris.config import (
    DATASET,
    DEBUG_DIR,
    NORMALIZED_DIR,
    REPORT_PATH,
    SAVE_DEBUG_IMAGES,
    SAVE_NORMALIZED_IMAGES
)

from src.iris.processor import (
    process_image
)

from src.iris.visualization import (
    create_debug_image
)


# ============================================================
# CSV COLUMNS
# ============================================================

FIELDNAMES = [

    "subject",
    "eye",
    "filename",

    "width",
    "height",

    "pupil_detected",
    "iris_detected",

    "pupil_x",
    "pupil_y",
    "pupil_radius",

    "iris_x",
    "iris_y",
    "iris_radius",

    "detection_status",
    "quality_status",

    "sharpness_laplacian",
    "iris_contrast_std",
    "iris_brightness_mean",

    "reflection_fraction",
    "dark_fraction",

    "lid_occlusion_fraction",
    "lash_fraction",

    "iris_area_fraction",

    "boundary_mean",
    "boundary_median",
    "boundary_support",
    "boundary_valid_samples",

    "pupil_iris_center_distance",
    "iris_pupil_radius_ratio"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("IRIS QUALITY CHECK")
    print("=" * 60)
    print()

    print(
        "Dataset:",
        DATASET
    )

    print(
        "Output:",
        REPORT_PATH.parent
    )

    print()

    # --------------------------------------------------------
    # Find subjects
    # --------------------------------------------------------

    subjects = [

        x
        for x in DATASET.iterdir()

        if (
            x.is_dir()
            and
            x.name.isdigit()
        )
    ]

    subjects.sort(
        key=lambda x: int(x.name)
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0
    successful = 0
    failed_pupil = 0
    failed_iris = 0
    read_errors = 0

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        for subject in subjects:

            subject_id = subject.name

            for eye in [
                "left",
                "right"
            ]:

                eye_folder = (
                    subject /
                    eye
                )

                if not eye_folder.exists():
                    continue

                images = sorted(
                    eye_folder.glob(
                        "*.bmp"
                    )
                )

                for image_path in images:

                    total += 1

                    (
                        normalized,
                        result,
                        pupil,
                        iris,
                        status
                    ) = process_image(
                        image_path,
                        subject_id,
                        eye
                    )

                    # ----------------------------------------
                    # Counters
                    # ----------------------------------------

                    if status == "DETECTED":

                        successful += 1

                    elif status == "FAILED_PUPIL":

                        failed_pupil += 1

                    elif status == "FAILED_IRIS":

                        failed_iris += 1

                    elif status == "READ_ERROR":

                        read_errors += 1

                    # ----------------------------------------
                    # CSV
                    # ----------------------------------------

                    writer.writerow(
                        result
                    )

                    # ----------------------------------------
                    # Output name
                    # ----------------------------------------

                    output_name = (
                        f"{subject_id}_"
                        f"{eye}_"
                        f"{image_path.stem}"
                    )

                    # ----------------------------------------
                    # Debug
                    # ----------------------------------------

                    if SAVE_DEBUG_IMAGES:

                        metrics = None

                        if (
                            result[
                                "sharpness_laplacian"
                            ] != ""
                        ):

                            metrics = {

                                key: float(
                                    result[key]
                                )

                                for key in [

                                    "sharpness_laplacian",
                                    "iris_contrast_std",
                                    "iris_brightness_mean",
                                    "reflection_fraction",
                                    "lid_occlusion_fraction",
                                    "boundary_mean",
                                    "boundary_support"
                                ]
                            }

                        gray = cv2.imread(
                            str(image_path),
                            cv2.IMREAD_GRAYSCALE
                        )

                        debug = (
                            create_debug_image(
                                gray,
                                pupil,
                                iris,
                                metrics,
                                status
                            )
                        )

                        debug_path = (
                            DEBUG_DIR /
                            f"{output_name}_debug.png"
                        )

                        cv2.imwrite(
                            str(debug_path),
                            debug
                        )

                    # ----------------------------------------
                    # Normalized output
                    # ----------------------------------------

                    if (
                        SAVE_NORMALIZED_IMAGES
                        and
                        normalized is not None
                    ):

                        normalized_path = (
                            NORMALIZED_DIR /
                            f"{output_name}.png"
                        )

                        cv2.imwrite(
                            str(
                                normalized_path
                            ),
                            normalized
                        )

                    print(
                        f"[{total:03d}] "
                        f"{subject_id}/{eye}/"
                        f"{image_path.name} | "
                        f"{status}"
                    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("QUALITY CHECK SUMMARY")
    print("=" * 60)

    print(
        f"Total images      : {total}"
    )

    print(
        f"Detected          : {successful}"
    )

    print(
        f"Failed pupil      : {failed_pupil}"
    )

    print(
        f"Failed iris       : {failed_iris}"
    )

    print(
        f"Read errors       : {read_errors}"
    )

    print()

    print(
        "CSV report:"
    )

    print(
        REPORT_PATH
    )

    print()

    print(
        "Debug images:"
    )

    print(
        DEBUG_DIR
    )

    print()

    print(
        "Normalized images:"
    )

    print(
        NORMALIZED_DIR
    )

    print()
    print("=" * 60)
    print(
        "NO QUALITY THRESHOLDS APPLIED"
    )
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()