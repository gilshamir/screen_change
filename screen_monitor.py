import argparse
from dataclasses import dataclass
from enum import IntEnum
import socket
import time

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import ImageGrab
from pynput import keyboard

BIG_PATTERN_RADIUS_FRACTION = 0.10
SMALL_PATTERN_RADIUS_FRACTION = 0.035
PATTERN_RADII = (28, 56, 196)
PATTERN_RADIUS_TOLERANCE = 0.10


class PatternId(IntEnum):
    TINY = 1
    SMALL = 2
    BIG = 3


@dataclass(frozen=True)
class PatternMatch:
    score: float
    center_x: int
    center_y: int
    radius: int


def _circle_boundary_templates(
    radius: int,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32]]:
    """
    Build one edge template for a candidate pattern size.

    The large template places four circle boundaries in a cross.  It is
    compared against a Sobel edge image, not against raw pixel colors.  The
    mask limits matching to the ring pixels instead of the entire bounding box.
    """
    size = 6 * radius + 1
    template = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    circle_template = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    ring_width = max(1, round(radius * 0.15))
    centers = (
        (3 * radius, radius),
        (radius, 3 * radius),
        (5 * radius, 3 * radius),
        (3 * radius, 5 * radius),
    )
    for center in centers:
        cv2.circle(template, center, radius, 1, thickness=ring_width)
        cv2.circle(mask, center, radius, 255, thickness=ring_width)
    cv2.circle(
        circle_template,
        (radius, radius),
        radius,
        1,
        thickness=ring_width,
    )
    return template, mask, circle_template


def _has_four_circle_boundaries(
    edge_frame: NDArray[np.float32],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    search_frame: NDArray[np.uint8],
    circle_template: NDArray[np.float32],
    pattern_location: tuple[int, int],
    radius: int,
    allow_textured: bool = False,
) -> bool:
    """
    Verify a template peak with local checks for all four circle boundaries.

    This is a separate, stricter pass after template matching: each circle
    must have strong radial edges, and at least three interiors must remain
    smooth.  That permits one Gabor-textured circle while rejecting an
    arbitrary collection of unrelated edges.
    """
    ring = circle_template > 0
    ring_area = int(np.count_nonzero(ring))
    y_coords, x_coords = np.mgrid[: 2 * radius + 1, : 2 * radius + 1]
    radial_x = (x_coords - radius).astype(np.float32)
    radial_y = (y_coords - radius).astype(np.float32)
    radial_length = np.maximum(np.sqrt(radial_x**2 + radial_y**2), 1.0)
    radial_x /= radial_length
    radial_y /= radial_length
    inner_radius = max(1, round(radius * 0.6))
    inner_circle = np.zeros_like(circle_template, dtype=np.uint8)
    cv2.circle(inner_circle, (radius, radius), inner_radius, 255, thickness=-1)
    inner = inner_circle > 0
    pattern_x, pattern_y = pattern_location
    smooth_circles = 0
    centers = (
        (3 * radius, radius),
        (radius, 3 * radius),
        (5 * radius, 3 * radius),
        (3 * radius, 5 * radius),
    )
    for center_x, center_y in centers:
        x = pattern_x + center_x - radius
        y = pattern_y + center_y - radius
        circle_edges = edge_frame[y : y + 2 * radius + 1, x : x + 2 * radius + 1][
            ring
        ]
        circle_gradient_x = gradient_x[
            y : y + 2 * radius + 1, x : x + 2 * radius + 1
        ][ring]
        circle_gradient_y = gradient_y[
            y : y + 2 * radius + 1, x : x + 2 * radius + 1
        ][ring]
        circle_pixels = search_frame[
            y : y + 2 * radius + 1, x : x + 2 * radius + 1
        ][inner]
        energy = float(np.sum(circle_edges))
        squared_energy = float(np.sum(circle_edges**2))
        support = energy / np.sqrt(ring_area * squared_energy + 1e-8)
        if float(np.std(circle_pixels)) <= 2.0:
            smooth_circles += 1
        radial_alignment = float(
            np.sum(
                np.abs(
                    circle_gradient_x * radial_x[ring]
                    + circle_gradient_y * radial_y[ring]
                )
            )
            / (energy + 1e-8)
        )
        support_threshold = 0.45 if allow_textured else 0.55
        mean_threshold = 0.50 if allow_textured else 0.50
        alignment_threshold = 0.25 if allow_textured else 0.75
        if (
            support < support_threshold
            or float(np.mean(circle_edges)) < mean_threshold
            or radial_alignment < alignment_threshold
        ):
            return False
    return allow_textured or smooth_circles >= 3


def detect_pattern(
    gray_frame: NDArray[np.uint8],
    threshold: float = 0.82,
    search_scale: float = 0.18,
    pattern_radii: tuple[int, ...] = PATTERN_RADII,
    pattern_radius_tolerance: float = PATTERN_RADIUS_TOLERANCE,
) -> PatternMatch | None:
    """
    Find four cross-arranged circles, tolerating one textured interior.

    Parameters
    ----------
    threshold:
        Minimum normalized template-match score from 0 to 1.  Lower values
        find weaker matches but create more false positives.
    search_scale:
        Fixed fraction of the original frame width and height used for the
        search.  This is one downscaled resolution, not a list of resolutions.
    pattern_radii:
        Exact full-resolution circle radii to test.  Each radius is scaled
        once to the search-frame resolution and checked within a tolerance.
    pattern_radius_tolerance:
        Relative radius tolerance used around each exact pattern radius.
        0.10 checks 10% below and above each known radius.

    The returned location and radius are converted back to the original
    frame's coordinate system.
    """
    if gray_frame.ndim != 2:
        raise ValueError("gray_frame must be a two-dimensional grayscale image")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0 and at most 1")
    if not 0.0 < search_scale <= 1.0:
        raise ValueError("search_scale must be greater than 0 and at most 1")
    if not pattern_radii or any(radius < 1 for radius in pattern_radii):
        raise ValueError("pattern_radii must contain only positive radii")
    if pattern_radius_tolerance < 0.0:
        raise ValueError("pattern_radius_tolerance must not be negative")

    original_height, original_width = gray_frame.shape
    # Search on a fixed lower-resolution copy for speed.  The final match is
    # scaled back to original pixels with separate X and Y scale factors.
    search_width = max(1, round(original_width * search_scale))
    search_height = max(1, round(original_height * search_scale))
    if search_width != original_width or search_height != original_height:
        search_frame = cv2.resize(
            gray_frame,
            (search_width, search_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        search_frame = gray_frame

    blurred_frame = cv2.GaussianBlur(search_frame, (0, 0), sigmaX=2.5)
    gradient_x = cv2.Sobel(blurred_frame, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_frame, cv2.CV_32F, 0, 1, ksize=3)
    edge_frame = cv2.magnitude(gradient_x, gradient_y)

    best: tuple[float, int, tuple[int, int], int] | None = None
    scale_x = search_width / original_width
    scale_y = search_height / original_height
    radius_scale = (scale_x + scale_y) / 2

    candidate_radii: dict[int, int] = {}
    for nominal_radius in pattern_radii:
        radius_delta = round(nominal_radius * pattern_radius_tolerance)
        for candidate_radius in range(
            max(1, nominal_radius - radius_delta),
            nominal_radius + radius_delta + 1,
        ):
            candidate_radii.setdefault(candidate_radius, nominal_radius)

    for full_radius, nominal_radius in sorted(candidate_radii.items()):
        radius = max(1, round(full_radius * radius_scale))
        if 6 * radius + 1 > search_width or 6 * radius + 1 > search_height:
            continue

        template, mask, circle_template = _circle_boundary_templates(radius)
        scores = cv2.matchTemplate(
            edge_frame,
            template,
            cv2.TM_CCORR_NORMED,
            mask=mask,
        )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        pattern_height = search_height - 6 * radius
        pattern_width = search_width - 6 * radius
        scores = scores[:pattern_height, :pattern_width]
        # Test up to five candidate peaks for this radius because the strongest
        # match may fail the local four-circle verification.
        for _ in range(5):
            _, score, _, location = cv2.minMaxLoc(scores)
            if score < threshold:
                break
            if _has_four_circle_boundaries(
                edge_frame,
                gradient_x,
                gradient_y,
                search_frame,
                circle_template,
                location,
                radius,
            ):
                if best is None or score > best[0]:
                    best = (score, nominal_radius, location, radius)
                break
            x, y = location
            exclusion = max(1, 2 * radius)
            # Suppress this candidate neighborhood before looking for another
            # peak for the same radius.
            scores[
                max(0, y - exclusion) : min(pattern_height, y + exclusion + 1),
                max(0, x - exclusion) : min(pattern_width, x + exclusion + 1),
            ] = 0.0

    if best is None or best[0] < threshold:
        return _detect_textured_gabor_cross(
            gray_frame,
            (55,),
            pattern_radius_tolerance,
            1.0,
        )

    score, full_radius, location, search_radius = best
    scale_x = original_width / search_width
    scale_y = original_height / search_height
    return PatternMatch(
        score=score,
        center_x=round((location[0] + 3 * search_radius) * scale_x),
        center_y=round((location[1] + 3 * search_radius) * scale_y),
        radius=full_radius,
    )


def _detect_textured_gabor_cross(
    gray_frame: NDArray[np.uint8],
    pattern_radii: tuple[int, ...],
    pattern_radius_tolerance: float,
    search_scale: float,
) -> PatternMatch | None:
    """Fallback detector for patterns where every circle contains Gabor texture."""
    border_height = max(1, gray_frame.shape[0] // 20)
    border_width = max(1, gray_frame.shape[1] // 20)
    border_pixels = np.concatenate(
        (
            gray_frame[:border_height].ravel(),
            gray_frame[-border_height:].ravel(),
            gray_frame[:, :border_width].ravel(),
            gray_frame[:, -border_width:].ravel(),
        )
    )
    background = float(np.median(border_pixels))
    for delta in (5, 8, 12, 20, 30):
        blob_mask = (
            np.abs(gray_frame.astype(np.float32) - background) >= delta
        ).astype(np.uint8)
        blob_mask = cv2.morphologyEx(
            blob_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        _, _, blob_stats, blob_centers = cv2.connectedComponentsWithStats(
            blob_mask,
            8,
        )
        components: list[tuple[float, float, int, int, int]] = []
        for index in range(1, len(blob_stats)):
            x, y, width, height, area = blob_stats[index]
            if (
                area >= 5000
                and 80 <= width <= gray_frame.shape[1] // 3
                and 80 <= height <= gray_frame.shape[0] // 3
            ):
                components.append(
                    (
                        float(blob_centers[index][0]),
                        float(blob_centers[index][1]),
                        int(width),
                        int(height),
                        int(area),
                    )
                )
        if len(components) != 4:
            continue

        top = min(components, key=lambda component: component[1])
        bottom = max(components, key=lambda component: component[1])
        left = min(components, key=lambda component: component[0])
        right = max(components, key=lambda component: component[0])

        vertical_center = round((top[1] + bottom[1]) / 2)
        horizontal_center = round((left[0] + right[0]) / 2)
        center_x = round((horizontal_center + (top[0] + bottom[0]) / 2) / 2)
        center_y = round((vertical_center + (left[1] + right[1]) / 2) / 2)

        estimated_radius = round(
            np.median(
                [
                    top[2],
                    top[3],
                    bottom[2],
                    bottom[3],
                    left[2],
                    left[3],
                    right[2],
                    right[3],
                ]
            )
            / 2
        )
        return PatternMatch(
            score=1.0,
            center_x=center_x,
            center_y=center_y,
            radius=estimated_radius,
        )
    original_height, original_width = gray_frame.shape
    search_width = max(1, round(original_width * search_scale))
    search_height = max(1, round(original_height * search_scale))
    search_frame = cv2.resize(
        gray_frame,
        (search_width, search_height),
        interpolation=cv2.INTER_AREA,
    ) if (search_width, search_height) != (original_width, original_height) else gray_frame

    blurred = cv2.GaussianBlur(search_frame, (0, 0), sigmaX=2.5)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edge_frame = cv2.magnitude(gradient_x, gradient_y)
    radius_scale = (search_width / original_width + search_height / original_height) / 2

    best: PatternMatch | None = None
    for nominal_radius in reversed(sorted(pattern_radii)):
        for nominal_full_radius in (nominal_radius,):
            radius = max(1, round(nominal_full_radius * radius_scale))
            if 6 * radius + 1 > search_width or 6 * radius + 1 > search_height:
                continue
            _, _, circle_template = _circle_boundary_templates(radius)
            step = max(4, radius // 2)
            for center_y in range(3 * radius, search_height - 3 * radius, step):
                for center_x in range(3 * radius, search_width - 3 * radius, step):
                    pattern_location = (
                        center_x - 3 * radius,
                        center_y - 3 * radius,
                    )
                    if _has_four_circle_boundaries(
                        edge_frame,
                        gradient_x,
                        gradient_y,
                        search_frame,
                        circle_template,
                        pattern_location,
                        radius,
                        allow_textured=True,
                    ):
                        scale_x = original_width / search_width
                        scale_y = original_height / search_height
                        match = PatternMatch(
                            score=1.0,
                            center_x=round(center_x * scale_x),
                            center_y=round(center_y * scale_y),
                            radius=nominal_radius,
                        )
                        if best is None:
                            best = match
    return best


def classify_pattern_id(
    match: PatternMatch,
    frame_shape: tuple[int, int],
) -> PatternId:
    radius_fraction = match.radius / min(frame_shape)
    if radius_fraction >= BIG_PATTERN_RADIUS_FRACTION:
        return PatternId.BIG
    if radius_fraction >= SMALL_PATTERN_RADIUS_FRACTION:
        return PatternId.SMALL
    return PatternId.TINY


def classify_pattern(match: PatternMatch, frame_shape: tuple[int, int]) -> str:
    return {
        PatternId.BIG: "Big",
        PatternId.SMALL: "Small",
        PatternId.TINY: "Tiny",
    }[classify_pattern_id(match, frame_shape)]


def send_tcp_message(host: str, port: int, message: str, timeout: float) -> None:
    payload = message.encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(payload)


def capture_gray_screen() -> NDArray[np.uint8]:
    image = ImageGrab.grab(all_screens=True).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the desktop for a four-circle pattern and notify a TCP server."
    )
    parser.add_argument("--host", required=True, help="TCP server hostname or IP address")
    parser.add_argument("--port", required=True, type=int, help="TCP server port (1-65535)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="boundary-match threshold from 0 to 1 (default: 0.82)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.01,
        help="minimum delay between screen scans in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--rearm-frames",
        type=int,
        default=1,
        help="consecutive misses before another detection can trigger (default: 1)",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be greater than 0 and at most 1")
    if args.interval <= 0.0:
        parser.error("--interval must be greater than 0")
    if args.rearm_frames < 1:
        parser.error("--rearm-frames must be positive")
    return args


def main() -> None:
    args = parse_args()
    print(f"Watching all screens; TCP notifications go to {args.host}:{args.port}.")
    active = False
    consecutive_misses = 0

    def on_press(key):
        send_tcp_message(
            args.host,
            args.port,
            f"E;1;Konan;1;;;;Gabor;-1;{str(key)}\r\n",
            timeout=5.0,
        )

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            started = time.monotonic()
            gray_frame = capture_gray_screen()
            match = detect_pattern(gray_frame, threshold=args.threshold)
            if match is not None:
                consecutive_misses = 0
                if not active:
                    pattern_name = classify_pattern(match, gray_frame.shape)
                    pattern_id = classify_pattern_id(match, gray_frame.shape)
                    print(
                        f"{pattern_name} pattern detected (score {match.score:.3f}, "
                        f"center {match.center_x},{match.center_y}, radius {match.radius})."
                    )
                    send_tcp_message(
                        args.host,
                        args.port,
                        f"E;1;Konan;1;;;;Gabor;{pattern_id.value};{pattern_name}\r\n",
                        timeout=5.0,
                    )
                    print("TCP notification sent.")
                    active = True
            elif active:
                consecutive_misses += 1
                if consecutive_misses >= args.rearm_frames:
                    active = False

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, args.interval - elapsed))
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
