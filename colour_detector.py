"""Colour detection with plain image processing (HSV colour classification, no machine learning).

Steps for every frame:
 1. shrink + blur the frame, convert BGR -> HSV
 2. give every pixel a colour name using a ready-made lookup table
 3. build a black/white mask for each colour and clean it (morphological opening/closing)
 4. find contours and turn big enough ones into boxes
"""
import cv2
import numpy as np

import settings
from detections import Detection

COLOUR_ID = {name: number for number, name in enumerate(settings.COLOUR_NAMES)}
IGNORE_PIXEL = 255   # colour id meaning "do not use this pixel" (used for faces)


def saturation_limit_for(sensitivity):
    """Sensitivity 5 -> 45. Higher sensitivity = lower limit = paler colours count."""
    return 95 - 10 * sensitivity


def name_the_colour(hue, saturation, brightness, saturation_limit=45):
    """Turn one HSV value (hue 0-179, saturation 0-255, brightness 0-255) into a colour name.
    Red wraps around hue 0, so it is checked with 'degrees < 15 or degrees >= 345'."""
    if brightness < settings.BLACK_BELOW_BRIGHTNESS:
        return "Black"
    if saturation < saturation_limit:
        return "White" if brightness >= settings.WHITE_ABOVE_BRIGHTNESS else "Gray"

    degrees = hue * 2   # OpenCV hue is 0-179, a full circle is 360 degrees
    if degrees < 15 or degrees >= 345:
        if brightness < 120:
            return "Maroon"
        return "Pink" if (saturation < 120 and brightness > 180) else "Red"
    if degrees < 40:
        if saturation < 110 and brightness > 180:
            return "Beige"
        return "Brown" if brightness < 150 else "Orange"
    if degrees < 70:
        return "Olive" if brightness < 150 else "Yellow"
    if degrees < 95:
        return "Lime"
    if degrees < 165:
        return "Dark Green" if brightness < 110 else "Green"
    if degrees < 195:
        return "Teal" if brightness < 130 else "Cyan"
    if degrees < 255:
        if brightness < 120:
            return "Navy"
        return "Light Blue" if (saturation < 140 and brightness > 190) else "Blue"
    if degrees < 290:
        return "Lavender" if (saturation < 110 and brightness > 200) else "Purple"
    if degrees < 330:
        return "Magenta"
    return "Pink"


def build_colour_lookup(sensitivity=settings.DEFAULT_SENSITIVITY):
    """Pre-compute a colour id for every hue/saturation/brightness combination.
    Saturation and brightness are grouped in steps of 8 (32 levels each) to keep it small.
    Later, every video frame needs only one fast table lookup."""
    limit = saturation_limit_for(sensitivity)
    lookup = np.zeros((180, 32, 32), np.uint8)
    for hue in range(180):
        for saturation_level in range(32):
            for brightness_level in range(32):
                name = name_the_colour(hue, saturation_level * 8 + 4, brightness_level * 8 + 4, limit)
                lookup[hue, saturation_level, brightness_level] = COLOUR_ID[name]
    return lookup


def balance_white(frame):
    """'Gray-world' white balance: scale B, G, R so their averages match.
    Helps under yellow room light. Can be wrong if one colour fills the whole frame."""
    channel_means = frame.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = channel_means.mean() / channel_means
    return np.clip(frame * gains, 0, 255).astype(np.uint8)


def face_area_to_ignore(face):
    """The face box made a bit bigger, as (left, top, right, bottom). Skin and hair are not 'colours'."""
    extra_x = int(face.width * settings.FACE_EXTRA_MARGIN)
    extra_y = int(face.height * settings.FACE_EXTRA_MARGIN)
    return (face.left - extra_x, face.top - extra_y, face.right + extra_x, face.bottom + extra_y)


def find_colour_regions(frame, lookup, ignore_areas=(), colours_to_find=None,
                        show_neutrals=True, use_white_balance=False):
    """Find coloured regions in a frame.
    ignore_areas = rectangles (left, top, right, bottom) that are skipped, e.g. faces.
    Returns: (regions, mask, colour_shares)
      regions        - list of Detection, largest first
      mask           - black/white image of all matching pixels (for the mask window)
      colour_shares  - {colour name: percent of the frame it covers}"""
    frame_width = frame.shape[1]
    shrink = min(1.0, settings.COLOUR_SEARCH_WIDTH / frame_width)
    small_frame = frame
    if shrink < 1.0:
        small_frame = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    small_frame = cv2.GaussianBlur(small_frame, (settings.BLUR_AMOUNT, settings.BLUR_AMOUNT), 0)
    if use_white_balance:
        small_frame = balance_white(small_frame)

    hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    colour_map = lookup[hsv_frame[:, :, 0], hsv_frame[:, :, 1] >> 3, hsv_frame[:, :, 2] >> 3]

    # Skip the ignored areas (faces, empty borders around an uploaded image)
    for left, top, right, bottom in ignore_areas:
        colour_map[max(0, int(top * shrink)):max(0, int(bottom * shrink)),
                   max(0, int(left * shrink)):max(0, int(right * shrink))] = IGNORE_PIXEL

    cleanup_kernel = np.ones((settings.CLEANUP_SIZE, settings.CLEANUP_SIZE), np.uint8)
    total_pixels = colour_map.shape[0] * colour_map.shape[1]
    smallest_area = settings.SMALLEST_REGION_FRACTION * total_pixels
    combined_mask = np.zeros(colour_map.shape, np.uint8)
    found = []           # (area, Detection)
    colour_shares = {}

    for colour_name in (colours_to_find or settings.COLOUR_NAMES):
        if colour_name in settings.NEUTRAL_COLOURS and not show_neutrals:
            continue
        mask = (colour_map == COLOUR_ID[colour_name]).astype(np.uint8) * 255   # binary mask
        if cv2.countNonZero(mask) < smallest_area:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)    # remove small specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleanup_kernel)   # fill small holes
        combined_mask |= mask

        share = 100.0 * cv2.countNonZero(mask) / total_pixels
        if share >= 0.5:
            colour_shares[colour_name] = share

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            left, top, width, height = cv2.boundingRect(contour)
            if (area >= smallest_area and width >= settings.THINNEST_REGION_PIXELS
                    and height >= settings.THINNEST_REGION_PIXELS):
                box = Detection(int(left / shrink), int(top / shrink),
                                int(width / shrink), int(height / shrink), colour_name)
                found.append((area, box))

    found.sort(key=lambda item: item[0], reverse=True)   # largest first
    regions = [box for _, box in found[:settings.MAX_BOXES]]
    return regions, combined_mask, colour_shares
