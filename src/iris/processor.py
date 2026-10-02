import cv2

from .pupil import detect_pupil
from .detector import detect_iris
from .quality import (
    calculate_quality_metrics,
    classify_quality
)
from .normalizer import normalize_iris


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    subject_id,
    eye
):

    result = {

        "subject": subject_id,
        "eye": eye,
        "filename": image_path.name,

        "width": "",
        "height": "",

        "pupil_detected": 0,
        "iris_detected": 0,

        "pupil_x": "",
        "pupil_y": "",
        "pupil_radius": "",

        "iris_x": "",
        "iris_y": "",
        "iris_radius": "",

        "detection_status": "",
        "quality_status":
            "NOT_CLASSIFIED",

        "sharpness_laplacian": "",
        "iris_contrast_std": "",
        "iris_brightness_mean": "",

        "reflection_fraction": "",
        "dark_fraction": "",

        "lid_occlusion_fraction": "",
        "lash_fraction": "",

        "iris_area_fraction": "",

        "boundary_mean": "",
        "boundary_median": "",
        "boundary_support": "",
        "boundary_valid_samples": "",

        "pupil_iris_center_distance": "",
        "iris_pupil_radius_ratio": ""
    }

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:

        result[
            "detection_status"
        ] = "READ_ERROR"

        return (
            None,
            result,
            None,
            None,
            "READ_ERROR"
        )

    h, w = gray.shape

    result["width"] = w
    result["height"] = h

    # --------------------------------------------------------
    # Pupil
    # --------------------------------------------------------

    pupil = detect_pupil(
        gray
    )

    if pupil is None:

        result[
            "detection_status"
        ] = "FAILED_PUPIL"

        return (
            None,
            result,
            None,
            None,
            "FAILED_PUPIL"
        )

    result[
        "pupil_detected"
    ] = 1

    px, py, pr = pupil

    result[
        "pupil_x"
    ] = round(px, 3)

    result[
        "pupil_y"
    ] = round(py, 3)

    result[
        "pupil_radius"
    ] = round(pr, 3)

    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    iris = detect_iris(
        gray,
        pupil
    )

    if iris is None:

        result[
            "detection_status"
        ] = "FAILED_IRIS"

        return (
            None,
            result,
            pupil,
            None,
            "FAILED_IRIS"
        )

    result[
        "iris_detected"
    ] = 1

    ix, iy, ir = iris

    result[
        "iris_x"
    ] = round(ix, 3)

    result[
        "iris_y"
    ] = round(iy, 3)

    result[
        "iris_radius"
    ] = round(ir, 3)

    result[
        "detection_status"
    ] = "DETECTED"

    # --------------------------------------------------------
    # Quality metrics
    # --------------------------------------------------------

    metrics = calculate_quality_metrics(
        gray,
        pupil,
        iris
    )

    for key, value in metrics.items():

        result[key] = round(
            float(value),
            6
        )

        result["quality_status"] = classify_quality(
            metrics
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = normalize_iris(
        gray,
        pupil,
        iris
    )

    return (
        normalized,
        result,
        pupil,
        iris,
        "DETECTED"
    )