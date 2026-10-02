import cv2
import numpy as np

from .config import (
    IRIS_MIN_RADIUS,
    IRIS_MAX_RADIUS,
    PUPIL_IRIS_CENTER_RANGE
)


# ============================================================
# IRIS DETECTION
# ============================================================

def detect_iris(
    gray,
    pupil
):

    if pupil is None:
        return None

    px, py, pr = pupil

    h, w = gray.shape

    # --------------------------------------------------------
    # Enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    blurred = cv2.medianBlur(
        enhanced,
        7
    )

    # --------------------------------------------------------
    # Lash mask
    # --------------------------------------------------------

    horizontal_lashes = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 3)
        )
    )

    vertical_lashes = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 9)
        )
    )

    lash_response = cv2.max(
        horizontal_lashes,
        vertical_lashes
    )

    lash_mask = (
        lash_response >= 16
    )

    # --------------------------------------------------------
    # Radius range
    # --------------------------------------------------------

    min_radius = IRIS_MIN_RADIUS

    max_radius = min(
        int(
            min(h, w) * 0.45
        ),
        IRIS_MAX_RADIUS
    )

    if max_radius < min_radius:
        return None

    # --------------------------------------------------------
    # Hough circles
    # --------------------------------------------------------

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=30,
        param2=15,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    center_range = (
        PUPIL_IRIS_CENTER_RANGE
    )

    candidate_set = set()

    for cx in range(
        max(
            0,
            int(px) -
            center_range
        ),
        min(
            w,
            int(px) +
            center_range +
            1
        ),
        2
    ):

        for cy in range(
            max(
                0,
                int(py) -
                center_range
            ),
            min(
                h,
                int(py) +
                center_range +
                1
            ),
            2
        ):

            if np.hypot(
                cx - px,
                cy - py
            ) > center_range:
                continue

            for radius in range(
                min_radius,
                max_radius + 1,
                2
            ):

                candidate_set.add(
                    (
                        cx,
                        cy,
                        radius
                    )
                )

    # --------------------------------------------------------
    # Add Hough candidates
    # --------------------------------------------------------

    if circles is not None:

        circles = np.round(
            circles[0]
        ).astype(int)

        for x, y, r in circles:

            if (
                min_radius <= r <= max_radius
                and
                np.hypot(
                    x - px,
                    y - py
                ) <= center_range
            ):

                candidate_set.add(
                    (
                        int(x),
                        int(y),
                        int(r)
                    )
                )

    if not candidate_set:
        return None

    # --------------------------------------------------------
    # Boundary angles
    # --------------------------------------------------------

    angles = np.radians(
        [
            -35, -25, -15, -5,
             5, 15, 25, 35,
            145, 155, 165, 175,
            185, 195, 205, 215
        ]
    )

    best_candidate = None
    best_score = float(
        "-inf"
    )

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    for cx, cy, radius in candidate_set:

        edge_differences = []

        for angle in angles:

            inner_x = int(
                round(
                    cx +
                    (radius - 3) *
                    np.cos(angle)
                )
            )

            inner_y = int(
                round(
                    cy +
                    (radius - 3) *
                    np.sin(angle)
                )
            )

            outer_x = int(
                round(
                    cx +
                    (radius + 3) *
                    np.cos(angle)
                )
            )

            outer_y = int(
                round(
                    cy +
                    (radius + 3) *
                    np.sin(angle)
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

            if (
                lash_mask[
                    inner_y,
                    inner_x
                ]
                or
                lash_mask[
                    outer_y,
                    outer_x
                ]
            ):
                continue

            edge_differences.append(
                float(
                    enhanced[
                        outer_y,
                        outer_x
                    ]
                )
                -
                float(
                    enhanced[
                        inner_y,
                        inner_x
                    ]
                )
            )

        if (
            len(edge_differences)
            <
            len(angles) // 2
        ):
            continue

        edge_differences.sort()

        robust_edge_score = float(
            np.mean(
                edge_differences[2:-2]
            )
        )

        center_offset = np.hypot(
            cx - px,
            cy - py
        )

        radius_prior = abs(
            radius -
            2.35 * pr
        )

        score = (
            robust_edge_score
            -
            0.45 * center_offset
            -
            0.25 * radius_prior
        )

        if score > best_score:

            best_score = score

            best_candidate = (
                float(cx),
                float(cy),
                float(radius)
            )

    return best_candidate