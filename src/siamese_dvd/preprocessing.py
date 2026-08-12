"""Classical CV preprocessing for DVD-cover localization and rectification."""

from pathlib import Path

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    points = np.asarray(points, dtype=np.float32)

    if points.shape != (4, 2):
        raise ValueError(
            f"Expected four 2D points, received shape {points.shape}."
        )

    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]

    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]

    return ordered


def four_point_transform(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Rectify a quadrilateral region into a top-down rectangular view."""

    top_left, top_right, bottom_right, bottom_left = order_points(
        points
    )

    width_a = np.linalg.norm(
        bottom_right - bottom_left
    )
    width_b = np.linalg.norm(
        top_right - top_left
    )

    max_width = max(
        int(round(width_a)),
        int(round(width_b)),
    )

    height_a = np.linalg.norm(
        top_right - bottom_right
    )
    height_b = np.linalg.norm(
        top_left - bottom_left
    )

    max_height = max(
        int(round(height_a)),
        int(round(height_b)),
    )

    if max_width < 2 or max_height < 2:
        raise ValueError(
            "Detected quadrilateral is too small to rectify."
        )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(
        np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype=np.float32,
        ),
        destination,
    )

    return cv2.warpPerspective(
        image,
        transform_matrix,
        (max_width, max_height),
    )

def quadrilateral_geometry(
    points: np.ndarray,
) -> tuple[float, float, float]:
    """Return estimated width, height, and long/short aspect ratio."""

    top_left, top_right, bottom_right, bottom_left = order_points(
        points
    )

    width_a = np.linalg.norm(
        bottom_right - bottom_left
    )
    width_b = np.linalg.norm(
        top_right - top_left
    )

    height_a = np.linalg.norm(
        top_right - bottom_right
    )
    height_b = np.linalg.norm(
        top_left - bottom_left
    )

    width = max(width_a, width_b)
    height = max(height_a, height_b)

    short_side = min(width, height)
    long_side = max(width, height)

    if short_side <= 0:
        return width, height, float("inf")

    aspect_ratio = long_side / short_side

    return width, height, aspect_ratio

def find_dvd_quadrilaterals(
    image: np.ndarray,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.95,
    min_aspect_ratio: float = 1.0,
    max_aspect_ratio: float = 2.5,
) -> list[np.ndarray]:
    """Return plausible DVD-cover quadrilaterals ordered by contour area.

    Multiple edge-detection settings are evaluated to improve robustness
    to lighting, glare, weak boundaries, and camera differences.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    canny_thresholds = [
        (50, 150),
        (75, 200),
        (100, 250),
    ]

    approximation_ratios = [
        0.02,
        0.03,
        0.04,
    ]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    candidate_records = []
    seen_boxes = set()

    for canny_low, canny_high in canny_thresholds:
        edges = cv2.Canny(
            blurred,
            canny_low,
            canny_high,
        )

        # Reconnect broken edges so the DVD boundary is more likely
        # to form a closed contour.
        closed_edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            closed_edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            area_ratio = contour_area / image_area

            if not (
                min_area_ratio
                <= area_ratio
                <= max_area_ratio
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            for approximation_ratio in approximation_ratios:
                approximation = cv2.approxPolyDP(
                    contour,
                    approximation_ratio * perimeter,
                    True,
                )

                if len(approximation) != 4:
                    continue

                if not cv2.isContourConvex(
                    approximation
                ):
                    continue

                corners = approximation.reshape(
                    4,
                    2,
                )

                _, _, aspect_ratio = quadrilateral_geometry(
                    corners
                )

                if not (
                    min_aspect_ratio
                    <= aspect_ratio
                    <= max_aspect_ratio
                ):
                    continue

                x, y, width, height = cv2.boundingRect(
                    approximation
                )

                # Avoid returning the same contour repeatedly from
                # different Canny/approximation settings.
                box_key = (
                    round(x / 10),
                    round(y / 10),
                    round(width / 10),
                    round(height / 10),
                )

                if box_key in seen_boxes:
                    break

                seen_boxes.add(box_key)

                candidate_records.append(
                    (
                        contour_area,
                        corners,
                    )
                )

                break

    candidate_records.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        corners
        for _, corners in candidate_records
    ]

def detect_dvd_quadrilateral(
    image: np.ndarray,
) -> np.ndarray | None:
    """Return the largest plausible DVD-cover quadrilateral."""

    candidates = find_dvd_quadrilaterals(
        image
    )

    if not candidates:
        return None

    return candidates[0]

def is_valid_rectified_crop(
    image: np.ndarray,
    min_side: int = 80,
    min_aspect_ratio: float = 1.15,
    max_aspect_ratio: float = 1.85,
    min_intensity_std: float = 8.0,
    max_edge_density: float = 10.0,
) -> bool:
    """Check whether a rectified candidate resembles a usable DVD cover."""

    if image is None or image.size == 0:
        return False

    height, width = image.shape[:2]

    if min(height, width) < min_side:
        return False

    short_side = min(height, width)
    long_side = max(height, width)

    aspect_ratio = long_side / short_side

    if not (
        min_aspect_ratio
        <= aspect_ratio
        <= max_aspect_ratio
    ):
        return False

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    intensity_std = float(
        gray.std()
    )

    if intensity_std < min_intensity_std:
        return False

    edges = cv2.Canny(
        gray,
        75,
        200,
    )

    edge_density = (
        np.count_nonzero(edges)
        / edges.size
        * 100
    )

    if edge_density > max_edge_density:
        return False

    return True

def rectify_dvd_cover(
    image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find and rectify the first valid DVD-cover candidate.

    Candidate quadrilaterals are considered from largest to smallest.
    Invalid geometry, implausible aspect ratio, and low-texture crops
    are rejected before trying the next candidate.
    """

    candidates = find_dvd_quadrilaterals(
        image
    )

    if not candidates:
        return None, None

    for corners in candidates:
        try:
            rectified = four_point_transform(
                image,
                corners,
            )
        except ValueError:
            continue

        if not is_valid_rectified_crop(
            rectified
        ):
            continue

        return rectified, corners

    return None, None


def preprocess_image_file(
    image_path: str | Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a captured image and attempt DVD-cover rectification."""

    image_path = Path(image_path)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return rectify_dvd_cover(image)