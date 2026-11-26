import socket
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from screen_monitor import (
    PatternId,
    PatternMatch,
    classify_pattern,
    classify_pattern_id,
    detect_pattern,
    send_tcp_message,
)


def make_pattern(
    radius: int,
    intensity: int,
    include_bottom_circle: bool = True,
    gabor_circle: int | None = None,
    gabor_amplitude: int = 70,
    gabor_orientation: float = np.pi / 4,
) -> np.ndarray:
    image = np.full((240, 320), 158, dtype=np.uint8)
    center_x, center_y = 160, 120
    centers = [
        (center_x, center_y - 2 * radius),
        (center_x - 2 * radius, center_y),
        (center_x + 2 * radius, center_y),
    ]
    if include_bottom_circle:
        centers.append((center_x, center_y + 2 * radius))
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    for index, center in enumerate(centers):
        cx, cy = center
        inside_circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        if index == gabor_circle:
            phase = (x - cx) * np.cos(gabor_orientation) + (y - cy) * np.sin(
                gabor_orientation
            )
            envelope = np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * (radius * 0.6) ** 2)
            )
            gabor = 158 + intensity + gabor_amplitude * np.cos(
                2 * np.pi * phase / 8
            ) * envelope
            image[inside_circle] = np.clip(gabor[inside_circle], 0, 255).astype(
                np.uint8
            )
        else:
            image[inside_circle] = 158 + intensity
    return image


def make_large_cross_pattern(radius: int, intensity: int) -> np.ndarray:
    image = np.full((1200, 1920), 158, dtype=np.uint8)
    center_x, center_y = 960, 600
    centers = (
        (center_x, center_y - 2 * radius),
        (center_x - 2 * radius, center_y),
        (center_x + 2 * radius, center_y),
        (center_x, center_y + 2 * radius),
    )
    for cx, cy in centers:
        cv2.circle(image, (cx, cy), radius, 158 + intensity, thickness=-1)
    return image


class DetectPatternTests(unittest.TestCase):
    def test_detects_different_sizes_and_intensities(self) -> None:
        for radius, intensity in ((20, 3), (36, 10)):
            with self.subTest(radius=radius, intensity=intensity):
                match = detect_pattern(
                    make_pattern(radius, intensity),
                    threshold=0.82,
                    search_scale=0.5,
                    pattern_radii=(20, 36),
                )
                self.assertIsNotNone(match)
                assert match is not None
                self.assertAlmostEqual(match.center_x, 160, delta=2)
                self.assertAlmostEqual(match.center_y, 120, delta=2)
                self.assertAlmostEqual(match.radius, radius, delta=2)

    def test_accepts_radius_within_nominal_tolerance(self) -> None:
        match = detect_pattern(
            make_large_cross_pattern(52, 8),
            threshold=0.82,
            search_scale=0.18,
            pattern_radii=(56,),
            pattern_radius_tolerance=0.15,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(match.center_x, 960, delta=15)
        self.assertAlmostEqual(match.center_y, 600, delta=15)
        self.assertEqual(match.radius, 56)

    def test_rejects_plain_background(self) -> None:
        background = np.full((240, 320), 158, dtype=np.uint8)
        self.assertIsNone(
            detect_pattern(
                background,
                search_scale=0.5,
                pattern_radii=(28,),
            )
        )

    def test_rejects_pattern_missing_one_circle(self) -> None:
        self.assertIsNone(
            detect_pattern(
                make_pattern(28, 8, include_bottom_circle=False),
                threshold=0.82,
                search_scale=0.5,
                pattern_radii=(28,),
            )
        )

    def test_detects_gabor_texture_in_any_circle(self) -> None:
        for gabor_circle in range(4):
            with self.subTest(gabor_circle=gabor_circle):
                match = detect_pattern(
                    make_pattern(
                        36,
                        6,
                        gabor_circle=gabor_circle,
                        gabor_orientation=gabor_circle * np.pi / 4,
                    ),
                    threshold=0.82,
                    search_scale=0.5,
                    pattern_radii=(36,),
                )
                self.assertIsNotNone(match)
                assert match is not None
                self.assertAlmostEqual(match.center_x, 160, delta=2)
                self.assertAlmostEqual(match.center_y, 120, delta=2)
                self.assertAlmostEqual(match.radius, 36, delta=2)

    def test_detects_supplied_gabor_sample(self) -> None:
        gabor_path = Path(__file__).with_name("BigGabor.png")
        if not gabor_path.exists():
            self.skipTest("BigGabor.png fixture is not present")
        image = cv2.imread(
            str(gabor_path),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(image)
        assert image is not None
        match = detect_pattern(image)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(match.center_x, 960, delta=15)
        self.assertAlmostEqual(match.center_y, 600, delta=15)
        self.assertAlmostEqual(match.radius, 196, delta=15)
        self.assertEqual(classify_pattern(match, image.shape), "Big")

    def test_detects_supplied_big_sample(self) -> None:
        image = cv2.imread(
            str(Path(__file__).with_name("Big.png")),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(image)
        assert image is not None
        match = detect_pattern(image)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(match.center_x, 960, delta=15)
        self.assertAlmostEqual(match.center_y, 600, delta=15)
        self.assertEqual(match.radius, 196)
        self.assertEqual(classify_pattern(match, image.shape), "Big")

    def test_classifies_supplied_pattern_sizes(self) -> None:
        screen_shape = (1200, 1920)
        for radius, expected_id, expected_name in (
            (196, PatternId.BIG, "Big"),
            (56, PatternId.SMALL, "Small"),
            (28, PatternId.TINY, "Tiny"),
        ):
            with self.subTest(radius=radius):
                match = PatternMatch(0.94, 960, 600, radius)
                self.assertEqual(classify_pattern_id(match, screen_shape), expected_id)
                self.assertEqual(classify_pattern(match, screen_shape), expected_name)

    def test_detects_supplied_tiny_sample(self) -> None:
        image = cv2.imread(
            str(Path(__file__).with_name("Tiny.png")),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(image)
        assert image is not None
        match = detect_pattern(image)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(match.center_x, 960, delta=15)
        self.assertAlmostEqual(match.center_y, 600, delta=15)
        self.assertLess(match.radius, 56)
        self.assertEqual(classify_pattern(match, image.shape), "Tiny")

    def test_detects_supplied_all_gabor_sample(self) -> None:
        image = cv2.imread(
            str(Path(__file__).with_name("Gabors.png")),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(image)
        assert image is not None
        match = detect_pattern(image)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertAlmostEqual(match.center_x, 960, delta=30)
        self.assertAlmostEqual(match.center_y, 600, delta=30)
        self.assertAlmostEqual(match.radius, 155, delta=15)
        self.assertEqual(classify_pattern(match, image.shape), "Big")

    def test_sends_newline_terminated_tcp_message(self) -> None:
        received: list[bytes] = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(2.0)
            port = listener.getsockname()[1]

            def receive_message() -> None:
                connection, _ = listener.accept()
                with connection:
                    received.append(connection.recv(1024))

            receiver = threading.Thread(target=receive_message)
            receiver.start()
            send_tcp_message(
                "127.0.0.1",
                port,
                "E;1;Konan;1;;;;Gabor;3;Big\r\n",
                timeout=2.0,
            )
            receiver.join(timeout=2.0)

        self.assertFalse(receiver.is_alive())
        self.assertEqual(received, [b"E;1;Konan;1;;;;Gabor;3;Big\r\n"])


if __name__ == "__main__":
    unittest.main()
