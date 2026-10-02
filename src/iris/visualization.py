import cv2


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def create_debug_image(
    image,
    pupil,
    iris,
    metrics,
    status
):

    if image.ndim == 2:

        debug = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    else:

        debug = image.copy()

    # --------------------------------------------------------
    # Pupil
    # --------------------------------------------------------

    if pupil is not None:

        px, py, pr = pupil

        cv2.circle(
            debug,
            (
                int(px),
                int(py)
            ),
            int(pr),
            (0, 255, 0),
            2
        )

        cv2.circle(
            debug,
            (
                int(px),
                int(py)
            ),
            2,
            (0, 255, 0),
            -1
        )

    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    if iris is not None:

        ix, iy, ir = iris

        cv2.circle(
            debug,
            (
                int(ix),
                int(iy)
            ),
            int(ir),
            (0, 0, 255),
            2
        )

        cv2.circle(
            debug,
            (
                int(ix),
                int(iy)
            ),
            2,
            (0, 0, 255),
            -1
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    cv2.putText(
        debug,
        f"STATUS: {status}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Pupil information
    # --------------------------------------------------------

    if pupil is not None:

        px, py, pr = pupil

        cv2.putText(
            debug,
            (
                f"Pupil: "
                f"({px:.1f}, {py:.1f}) "
                f"r={pr:.1f}"
            ),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # --------------------------------------------------------
    # Iris information
    # --------------------------------------------------------

    if iris is not None:

        ix, iy, ir = iris

        cv2.putText(
            debug,
            (
                f"Iris: "
                f"({ix:.1f}, {iy:.1f}) "
                f"r={ir:.1f}"
            ),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if metrics:

        lines = [
            (
                "Sharpness: "
                f"{metrics['sharpness_laplacian']:.1f}"
            ),
            (
                "Contrast: "
                f"{metrics['iris_contrast_std']:.1f}"
            ),
            (
                "Brightness: "
                f"{metrics['iris_brightness_mean']:.1f}"
            ),
            (
                "Boundary: "
                f"{metrics['boundary_mean']:.1f}"
            ),
            (
                "Boundary support: "
                f"{metrics['boundary_support']:.2f}"
            ),
            (
                "Occlusion: "
                f"{metrics['lid_occlusion_fraction']:.2f}"
            ),
            (
                "Reflection: "
                f"{metrics['reflection_fraction']:.2f}"
            )
        ]

        y = 95

        for line in lines:

            cv2.putText(
                debug,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            y += 18

    return debug