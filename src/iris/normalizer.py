import cv2
import numpy as np

from .config import (
    RADIAL_RESOLUTION,
    ANGULAR_RESOLUTION,
    NORMALIZED_HEIGHT,
    NORMALIZED_WIDTH
)

from .occlusion import (
    detect_lid_occlusion,
    detect_lash_mask,
    detect_reflection_mask
)


# ============================================================
# RAY-CIRCLE INTERSECTION
# ============================================================

def ray_circle_intersection(
    px,
    py,
    dx,
    dy,
    cx,
    cy,
    radius
):

    fx = px - cx
    fy = py - cy

    b = (
        fx * dx +
        fy * dy
    )

    c = (
        fx * fx +
        fy * fy -
        radius * radius
    )

    discriminant = (
        b * b -
        c
    )

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(
        discriminant
    )

    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    t = None

    if t1 > 0:
        t = t1

    if t2 > 0:

        if (
            t is None
            or
            t2 < t
        ):
            t = t2

    if t is None:
        return None

    return (
        px + t * dx,
        py + t * dy
    )


# ============================================================
# RUBBER-SHEET NORMALIZATION
# ============================================================

def normalize_iris(
    image,
    pupil,
    iris
):

    px, py, pr = pupil
    ix, iy, ir = iris

    h, w = image.shape[:2]

    gray = (
        image
        if image.ndim == 2
        else cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    )

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    iris_region = (
        (xx - ix) ** 2 +
        (yy - iy) ** 2
        <= ir ** 2
    )

    pupil_region = (
        (xx - px) ** 2 +
        (yy - py) ** 2
        <= pr ** 2
    )

    # --------------------------------------------------------
    # Reflection
    # --------------------------------------------------------

    reflection_mask = (
        detect_reflection_mask(
            gray,
            iris_region,
            pupil_region
        )
    )

    inpainted = cv2.inpaint(
        gray,
        reflection_mask,
        3,
        cv2.INPAINT_TELEA
    )

    # --------------------------------------------------------
    # Eyelid
    # --------------------------------------------------------

    lid_mask = detect_lid_occlusion(
        gray,
        iris
    )

    # --------------------------------------------------------
    # Lash
    # --------------------------------------------------------

    lash_response = detect_lash_mask(
        gray
    )

    iris_annulus = (
        iris_region &
        ~pupil_region
    )

    lid_cap_region = (
        np.abs(
            yy - iy
        )
        >=
        0.45 * ir
    )

    lash_mask = (
        lash_response
        &
        iris_annulus
        &
        lid_cap_region
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Combined noise
    # --------------------------------------------------------

    noise_mask = cv2.bitwise_or(
        lid_mask,
        cv2.bitwise_or(
            reflection_mask,
            lash_mask
        )
    )

    # --------------------------------------------------------
    # Coordinate maps
    # --------------------------------------------------------

    map_x = np.zeros(
        (
            RADIAL_RESOLUTION,
            ANGULAR_RESOLUTION
        ),
        dtype=np.float32
    )

    map_y = np.zeros(
        (
            RADIAL_RESOLUTION,
            ANGULAR_RESOLUTION
        ),
        dtype=np.float32
    )

    for theta_index in range(
        ANGULAR_RESOLUTION
    ):

        theta = (
            2.0 *
            np.pi *
            theta_index /
            ANGULAR_RESOLUTION
        )

        dx = np.cos(theta)
        dy = np.sin(theta)

        pupil_x = (
            px +
            pr * dx
        )

        pupil_y = (
            py +
            pr * dy
        )

        iris_point = (
            ray_circle_intersection(
                px,
                py,
                dx,
                dy,
                ix,
                iy,
                ir
            )
        )

        if iris_point is None:

            iris_x = (
                ix +
                ir * dx
            )

            iris_y = (
                iy +
                ir * dy
            )

        else:

            iris_x, iris_y = (
                iris_point
            )

        for radial_index in range(
            RADIAL_RESOLUTION
        ):

            rho = (
                radial_index /
                (
                    RADIAL_RESOLUTION - 1
                )
            )

            map_x[
                radial_index,
                theta_index
            ] = (
                (1.0 - rho) *
                pupil_x
                +
                rho *
                iris_x
            )

            map_y[
                radial_index,
                theta_index
            ] = (
                (1.0 - rho) *
                pupil_y
                +
                rho *
                iris_y
            )

    # --------------------------------------------------------
    # Remapping
    # --------------------------------------------------------

    normalized = cv2.remap(
        inpainted,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    normalized_mask = cv2.remap(
        255 - noise_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    normalized = cv2.resize(
        normalized,
        (
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    normalized_mask = cv2.resize(
        normalized_mask,
        (
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT
        ),
        interpolation=cv2.INTER_NEAREST
    )

    normalized[
        normalized_mask == 0
    ] = 0

    return normalized