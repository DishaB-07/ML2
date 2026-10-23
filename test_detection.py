"""Simple automated checks. Run from the project folder:  python -m unittest discover -s tests -v
They need no webcam."""
import unittest

import cv2
import numpy as np

import settings
from colour_detector import build_colour_lookup, face_area_to_ignore, find_colour_regions, name_the_colour
from detections import BoxSmoother, Detection
from input_sources import fit_image_to_screen

# One clear example (BGR) for each colour name
EXAMPLE_COLOURS = {
    "Red": (0, 0, 220), "Maroon": (0, 0, 110), "Orange": (0, 140, 255), "Brown": (20, 50, 90),
    "Yellow": (0, 230, 240), "Olive": (0, 110, 110), "Lime": (0, 230, 150), "Green": (0, 200, 0),
    "Dark Green": (0, 90, 20), "Teal": (110, 110, 0), "Cyan": (220, 220, 0),
    "Light Blue": (250, 200, 140), "Blue": (220, 60, 0), "Navy": (110, 20, 0),
    "Purple": (180, 40, 130), "Lavender": (250, 190, 220), "Magenta": (200, 0, 255),
    "Pink": (190, 140, 255), "Beige": (190, 220, 245), "White": (245, 245, 245),
    "Gray": (128, 128, 128), "Black": (10, 10, 10),
}


class ColourNamingTests(unittest.TestCase):
    def test_every_colour_name_is_recognised_in_a_picture(self):
        lookup = build_colour_lookup()
        for name, bgr in EXAMPLE_COLOURS.items():
            background = (235, 235, 235) if name in ("Gray", "Black") else ((10, 10, 10) if name == "White" else (60, 60, 60))
            frame = np.full((720, 1280, 3), background, np.uint8)
            cv2.rectangle(frame, (500, 200), (700, 420), bgr, -1)
            regions, _, _ = find_colour_regions(frame, lookup)
            names_near_box = [r.label for r in regions if 440 < r.left < 560 and 150 < r.top < 260]
            self.assertEqual(names_near_box, [name], f"{name} was not recognised correctly")

    def test_red_works_on_both_sides_of_hue_zero(self):
        self.assertEqual(name_the_colour(2, 255, 220), "Red")
        self.assertEqual(name_the_colour(176, 255, 220), "Red")

    def test_grey_and_black_and_white(self):
        self.assertEqual(name_the_colour(0, 0, 10), "Black")
        self.assertEqual(name_the_colour(0, 0, 128), "Gray")
        self.assertEqual(name_the_colour(0, 0, 250), "White")

    def test_no_false_band_between_touching_colours(self):
        lookup = build_colour_lookup()
        frame = np.full((720, 1280, 3), (60, 60, 60), np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 220), -1)
        cv2.rectangle(frame, (400, 200), (600, 400), (0, 200, 0), -1)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        regions, _, _ = find_colour_regions(frame, lookup, show_neutrals=False)
        self.assertEqual(sorted(r.label for r in regions), ["Green", "Red"])

    def test_face_area_is_ignored(self):
        lookup = build_colour_lookup()
        frame = np.full((720, 1280, 3), (60, 60, 60), np.uint8)
        cv2.rectangle(frame, (300, 200), (500, 400), (140, 170, 225), -1)   # skin-like block
        face = Detection(300, 200, 200, 200, "Human Face")
        regions, _, _ = find_colour_regions(frame, lookup, [face_area_to_ignore(face)], show_neutrals=False)
        self.assertEqual(regions, [])


class SmootherTests(unittest.TestCase):
    def test_box_appears_after_confirmation_and_disappears_later(self):
        smoother = BoxSmoother()
        box = Detection(100, 100, 50, 50, "Red")
        self.assertEqual(smoother.update([box]), [])                    # seen once: not drawn yet
        self.assertEqual(len(smoother.update([box])), 1)                # seen twice: now drawn
        for _ in range(settings.FRAMES_TO_FORGET + 1):
            result = smoother.update([])
        self.assertEqual(result, [])


class ImageFitTests(unittest.TestCase):
    def test_image_is_fitted_without_stretching(self):
        tall_image = np.full((1000, 500, 3), 200, np.uint8)
        screen, empty_areas = fit_image_to_screen(tall_image, 1280, 720)
        self.assertEqual(screen.shape, (720, 1280, 3))
        self.assertEqual(len(empty_areas), 2)      # empty borders on the left and right


if __name__ == "__main__":
    unittest.main()
