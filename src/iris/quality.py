import cv2
import numpy as np

from .config import BOUNDARY_SAMPLES
from .occlusion import detect_lid_occlusion


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_quality_metrics(
    gray,
    pupil,
    iris
):

    h, w = gray.shape

    px, py, pr = pupil
    ix, iy, ir = iris

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    # --------------------------------------------------------
    # Regions
    # --------------------------------------------------------

    pupil_mask = (
        (xx - px) ** 2 +
        (yy - py) ** 2
        <= pr ** 2
    )

    iris_mask = (
        (xx - ix) ** 2 +
        (yy - iy) ** 2
        <= ir ** 2
    )

    annulus_mask = (
        iris_mask &
        ~pupil_mask
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------------------------
    # Iris pixels
    # --------------------------------------------------------

    iris_pixels = gray[
        annulus_mask
    ]

    if iris_pixels.size == 0:

        contrast = 0.0
        brightness = 0.0

    else:

        contrast = float(
            np.std(
                iris_pixels
            )
        )

        brightness = float(
            np.mean(
                iris_pixels
            )
        )

    # --------------------------------------------------------
    # Reflection
    # --------------------------------------------------------

    bright_pixels = (
        gray >= 221
    )

    if np.any(annulus_mask):

        reflection_fraction = float(
            np.mean(
                bright_pixels[
                    annulus_mask
                ]
            )
        )

    else:

        reflection_fraction = 0.0

    # --------------------------------------------------------
    # Dark fraction
    # --------------------------------------------------------

    dark_pixels = (
        gray <= 30
    )

    if np.any(annulus_mask):

        dark_fraction = float(
            np.mean(
                dark_pixels[
                    annulus_mask
                ]
            )
        )

    else:

        dark_fraction = 0.0

    # --------------------------------------------------------
    # Eyelid occlusion
    # --------------------------------------------------------

    lid_mask = detect_lid_occlusion(
        gray,
        iris
    )

    if np.any(annulus_mask):

        lid_occlusion_fraction = float(
            np.mean(
                lid_mask[
                    annulus_mask
                ] > 0
            )
        )

    else:

        lid_occlusion_fraction = 0.0

    # --------------------------------------------------------
    # Lash fraction
    # --------------------------------------------------------

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    horizontal_lashes = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 3)
        )
    )

    vertical_lashes = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 11)
        )
    )

    lash_response = cv2.max(
        horizontal_lashes,
        vertical_lashes
    )

    lash_mask = (
        lash_response >= 30
    )

    if np.any(annulus_mask):

        lash_fraction = float(
            np.mean(
                lash_mask[
                    annulus_mask
                ]
            )
        )

    else:

        lash_fraction = 0.0

    # --------------------------------------------------------
    # Iris size
    # --------------------------------------------------------

    iris_area = (
        np.pi *
        ir *
        ir
    )

    iris_area_fraction = float(
        iris_area /
        max(
            h * w,
            1
        )
    )

    # --------------------------------------------------------
    # Boundary consistency
    # --------------------------------------------------------

    boundary_scores = []

    valid_samples = 0

    for theta in np.linspace(
        0,
        2 * np.pi,
        BOUNDARY_SAMPLES,
        endpoint=False
    ):

        dx = np.cos(theta)
        dy = np.sin(theta)

        inner_x = int(
            round(
                ix +
                (ir - 3) *
                dx
            )
        )

        inner_y = int(
            round(
                iy +
                (ir - 3) *
                dy
            )
        )

        outer_x = int(
            round(
                ix +
                (ir + 3) *
                dx
            )
        )

        outer_y = int(
            round(
                iy +
                (ir + 3) *
                dy
            )
        )

        if not (
            0 <= inner_x < w
            and
            0 <= inner_y < h
            and
            0 <= outer_x < w
            and
            0 <= outer_y < h
        ):
            continue

        inner_value = float(
            gray[
                inner_y,
                inner_x
            ]
        )

        outer_value = float(
            gray[
                outer_y,
                outer_x
            ]
        )

        difference = abs(
            outer_value -
            inner_value
        )

        boundary_scores.append(
            difference
        )

        valid_samples += 1

    if boundary_scores:

        boundary_mean = float(
            np.mean(
                boundary_scores
            )
        )

        boundary_median = float(
            np.median(
                boundary_scores
            )
        )

        boundary_support = float(
            np.mean(
                np.array(
                    boundary_scores
                ) >= 10
            )
        )

    else:

        boundary_mean = 0.0
        boundary_median = 0.0
        boundary_support = 0.0

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    center_distance = float(
        np.hypot(
            ix - px,
            iy - py
        )
    )

    radius_ratio = float(
        ir /
        max(
            pr,
            1e-6
        )
    )

    return {

        "sharpness_laplacian":
            sharpness,

        "iris_contrast_std":
            contrast,

        "iris_brightness_mean":
            brightness,

        "reflection_fraction":
            reflection_fraction,

        "dark_fraction":
            dark_fraction,

        "lid_occlusion_fraction":
            lid_occlusion_fraction,

        "lash_fraction":
            lash_fraction,

        "iris_area_fraction":
            iris_area_fraction,

        "boundary_mean":
            boundary_mean,

        "boundary_median":
            boundary_median,

        "boundary_support":
            boundary_support,

        "boundary_valid_samples":
            valid_samples,

        "pupil_iris_center_distance":
            center_distance,

        "iris_pupil_radius_ratio":
            radius_ratio
    }

def classify_quality(metrics):

    penalty = 0

    # Boundary
    if metrics["boundary_support"] < 0.50:
        penalty += 2
    elif metrics["boundary_support"] < 0.65:
        penalty += 1

    # Eyelashes
    if metrics["lash_fraction"] > 0.25:
        penalty += 2
    elif metrics["lash_fraction"] > 0.20:
        penalty += 1

    # Eyelids
    if metrics["lid_occlusion_fraction"] > 0.22:
        penalty += 2
    elif metrics["lid_occlusion_fraction"] > 0.15:
        penalty += 1

    # Reflection
    if metrics["reflection_fraction"] > 0.08:
        penalty += 2
    elif metrics["reflection_fraction"] > 0.03:
        penalty += 1

    # Contrast
    if metrics["iris_contrast_std"] < 18:
        penalty += 2
    elif metrics["iris_contrast_std"] < 25:
        penalty += 1

    if penalty <= 1:
        return "GOOD"

    if penalty <= 3:
        return "ACCEPTABLE"

    return "POOR"