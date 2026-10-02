import cv2
import numpy as np


# ============================================================
# EYELID OCCLUSION
# ============================================================

def detect_lid_occlusion(
    image,
    iris
):

    h, w = image.shape[:2]

    ix, iy, ir = iris

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(image)

    smoothed = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    ).astype(
        np.float32
    )

    vertical_gradient = np.abs(
        np.diff(
            smoothed,
            axis=0
        )
    )

    occlusion = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    left = max(
        0,
        int(
            np.floor(
                ix - ir
            )
        )
    )

    right = min(
        w,
        int(
            np.ceil(
                ix + ir + 1
            )
        )
    )

    for x in range(
        left,
        right
    ):

        dx = x - ix

        remaining = (
            ir * ir -
            dx * dx
        )

        if remaining <= 0:
            continue

        half_height = np.sqrt(
            remaining
        )

        upper_arc = (
            iy -
            half_height
        )

        lower_arc = (
            iy +
            half_height
        )

        # ----------------------------------------------------
        # Upper eyelid
        # ----------------------------------------------------

        upper_start = max(
            0,
            int(
                np.floor(
                    upper_arc
                )
            ) - 3
        )

        upper_end = min(
            h - 1,
            int(
                np.ceil(
                    upper_arc
                )
            ) + 20
        )

        if upper_end > upper_start:

            profile = (
                vertical_gradient[
                    upper_start:upper_end,
                    x
                ]
            )

            edge_y = (
                upper_start +
                int(
                    np.argmax(
                        profile
                    )
                )
            )

            if (
                profile[
                    edge_y -
                    upper_start
                ] >= 14
                and
                edge_y >
                upper_arc + 4
            ):

                occlusion[
                    max(
                        0,
                        int(
                            np.ceil(
                                upper_arc
                            )
                        )
                    ):
                    edge_y + 1,
                    x
                ] = 255

        # ----------------------------------------------------
        # Lower eyelid
        # ----------------------------------------------------

        lower_start = max(
            0,
            int(
                np.floor(
                    lower_arc
                )
            ) - 20
        )

        lower_end = min(
            h - 1,
            int(
                np.ceil(
                    lower_arc
                )
            ) + 3
        )

        if lower_end > lower_start:

            profile = (
                vertical_gradient[
                    lower_start:lower_end,
                    x
                ]
            )

            edge_y = (
                lower_start +
                int(
                    np.argmax(
                        profile
                    )
                )
            )

            if (
                profile[
                    edge_y -
                    lower_start
                ] >= 14
                and
                edge_y <
                lower_arc - 4
            ):

                occlusion[
                    edge_y:
                    min(
                        h,
                        int(
                            np.floor(
                                lower_arc
                            )
                        ) + 1
                    ),
                    x
                ] = 255

    return occlusion


# ============================================================
# LASH MASK
# ============================================================

def detect_lash_mask(
    gray,
    threshold=30
):

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    horizontal = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 3)
        )
    )

    vertical = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 11)
        )
    )

    response = cv2.max(
        horizontal,
        vertical
    )

    return (
        response >= threshold
    )


# ============================================================
# REFLECTION MASK
# ============================================================

def detect_reflection_mask(
    gray,
    iris_region,
    pupil_region
):

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

    bright_pixels = cv2.inRange(
        gray,
        221,
        255
    )

    reflection_mask = cv2.bitwise_and(
        bright_pixels,
        dark_neighborhood
    )

    region_mask = (
        iris_region |
        pupil_region
    ).astype(
        np.uint8
    ) * 255

    reflection_mask = cv2.bitwise_and(
        reflection_mask,
        region_mask
    )

    return reflection_mask