import cv2
import numpy as np

from .config import (
    PUPIL_MIN_RADIUS,
    PUPIL_MAX_RADIUS
)


# ============================================================
# PUPIL DETECTION
# ============================================================

def detect_pupil(gray):

    h, w = gray.shape

    # --------------------------------------------------------
    # Reflection removal
    # --------------------------------------------------------

    dark_seed = cv2.inRange(
        gray,
        0,
        90
    )

    dark_neighborhood = cv2.dilate(
        dark_seed,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (21, 21)
        )
    )

    bright_mask = cv2.inRange(
        gray,
        231,
        255
    )

    reflection_mask = cv2.bitwise_and(
        bright_mask,
        dark_neighborhood
    )

    clean_gray = cv2.inpaint(
        gray,
        reflection_mask,
        5,
        cv2.INPAINT_TELEA
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    enhanced = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    ).apply(clean_gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (7, 7),
        0
    )

    # --------------------------------------------------------
    # Hough circle detection
    # --------------------------------------------------------

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=18,
        param1=80,
        param2=20,
        minRadius=PUPIL_MIN_RADIUS,
        maxRadius=PUPIL_MAX_RADIUS
    )

    if circles is None:

        circles = np.empty(
            (0, 3),
            dtype=int
        )

    else:

        circles = np.round(
            circles[0]
        ).astype(int)

    candidates = set()

    dark_component_candidates = []

    # --------------------------------------------------------
    # Hough candidates
    # --------------------------------------------------------

    for x, y, r in circles:

        if x < 0 or x >= w:
            continue

        if y < 0 or y >= h:
            continue

        if x < 40 or x > w - 40:
            continue

        if y < 40 or y > h - 35:
            continue

        if (
            r < PUPIL_MIN_RADIUS
            or
            r > PUPIL_MAX_RADIUS
        ):
            continue

        candidates.add(
            (
                int(x),
                int(y),
                int(r)
            )
        )

    # --------------------------------------------------------
    # Dark connected components
    # --------------------------------------------------------

    for threshold in (
        40,
        50,
        60
    ):

        dark_regions = (
            clean_gray < threshold
        ).astype(np.uint8)

        count, _, stats, centroids = (
            cv2.connectedComponentsWithStats(
                dark_regions
            )
        )

        for component in range(
            1,
            count
        ):

            left, top, width, height, area = (
                stats[component]
            )

            if left == 0:
                continue

            if top == 0:
                continue

            if left + width >= w:
                continue

            if top + height >= h:
                continue

            aspect_ratio = (
                width /
                max(height, 1)
            )

            if not (
                0.65 <= aspect_ratio <= 1.5
            ):
                continue

            fill_ratio = (
                area /
                max(
                    width * height,
                    1
                )
            )

            if fill_ratio < 0.35:
                continue

            radius = int(
                round(
                    np.sqrt(
                        area / np.pi
                    )
                )
            )

            x, y = np.round(
                centroids[component]
            ).astype(int)

            if (
                PUPIL_MIN_RADIUS
                <= radius
                <= PUPIL_MAX_RADIUS
                and
                40 <= x <= w - 40
                and
                40 <= y <= h - 35
            ):

                candidates.add(
                    (
                        int(x),
                        int(y),
                        radius
                    )
                )

                dark_component_candidates.append(
                    (
                        int(x),
                        int(y),
                        radius,
                        int(area)
                    )
                )

    if not candidates:
        return None

    # --------------------------------------------------------
    # Lash detection
    # --------------------------------------------------------

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    horizontal_lashes = cv2.morphologyEx(
        clean_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 3)
        )
    )

    vertical_lashes = cv2.morphologyEx(
        clean_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 11)
        )
    )

    lash_mask = (
        cv2.max(
            horizontal_lashes,
            vertical_lashes
        ) >= 18
    )

    # --------------------------------------------------------
    # Candidate scoring
    # --------------------------------------------------------

    scored_candidates = []

    for x, y, r in candidates:

        distance_squared = (
            (xx - x) ** 2
            +
            (yy - y) ** 2
        )

        inner_mask = (
            distance_squared
            <=
            (0.55 * r) ** 2
        ) & ~lash_mask

        annulus_mask = (
            (
                distance_squared
                >=
                (1.05 * r) ** 2
            )
            &
            (
                distance_squared
                <=
                (1.3 * r) ** 2
            )
        )

        inner_pixels = clean_gray[
            inner_mask
        ]

        annulus_pixels = clean_gray[
            annulus_mask
        ]

        if (
            inner_pixels.size == 0
            or
            annulus_pixels.size == 0
        ):
            continue

        inner_median = float(
            np.median(
                inner_pixels
            )
        )

        annulus_median = float(
            np.median(
                annulus_pixels
            )
        )

        dark_fraction = float(
            np.mean(
                inner_pixels < 60
            )
        )

        component_support = max(
            (
                max(
                    0.0,
                    1.0 -
                    (
                        np.hypot(
                            x - component_x,
                            y - component_y
                        )
                        +
                        abs(
                            r -
                            component_radius
                        )
                    )
                    /
                    max(
                        component_radius,
                        1
                    )
                )
                *
                min(
                    component_area /
                    1500.0,
                    1.0
                )
                for (
                    component_x,
                    component_y,
                    component_radius,
                    component_area
                )
                in dark_component_candidates
            ),
            default=0.0
        )

        score = (
            2.0 * dark_fraction
            +
            (
                annulus_median -
                inner_median
            ) / 64.0
            +
            1.5 * component_support
            -
            0.02 *
            abs(
                r - 27
            )
        )

        scored_candidates.append(
            (
                score,
                x,
                y,
                r
            )
        )

    if not scored_candidates:
        return None

    _, x, y, r = max(
        scored_candidates,
        key=lambda item: item[0]
    )

    return (
        float(x),
        float(y),
        float(r)
    )