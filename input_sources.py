"""Where the pictures come from: the webcam or an uploaded image file."""
import sys

import cv2
import numpy as np


def open_camera(camera_number):
    """Open the webcam. On Windows the DirectShow backend is faster and more reliable."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        camera = cv2.VideoCapture(camera_number, backend)
        if camera.isOpened():
            return camera
        camera.release()
    return None


def ask_for_image_file():
    """Show the normal 'open file' window and return the chosen path (or None)."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        print("tkinter is not available. Start the app with:  python main.py --image path\\to\\picture.jpg")
        return None
    window = tkinter.Tk()
    window.withdraw()
    window.attributes("-topmost", True)    # make sure the dialog is not hidden behind the camera window
    path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
    window.destroy()
    return path or None


def read_image_file(path):
    """Read an image from disk. Works with spaces and non-English letters in the path."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)    # None if the file is not an image
    except OSError:
        return None


def fit_image_to_screen(image, screen_width, screen_height):
    """Shrink or enlarge the image to fit the screen without stretching it.
    Empty space is filled with dark colour, so the layout matches the live camera view.
    Returns (screen, empty_areas): empty_areas lists the dark borders as (left, top, right, bottom)."""
    image_height, image_width = image.shape[:2]
    zoom = min(screen_width / image_width, screen_height / image_height)
    new_width, new_height = max(1, int(image_width * zoom)), max(1, int(image_height * zoom))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_CUBIC)
    screen = np.full((screen_height, screen_width, 3), (35, 30, 28), np.uint8)
    left, top = (screen_width - new_width) // 2, (screen_height - new_height) // 2
    screen[top:top + new_height, left:left + new_width] = resized
    right, bottom = left + new_width, top + new_height
    empty_areas = [area for area in (
        (0, 0, left, screen_height), (right, 0, screen_width, screen_height),
        (0, 0, screen_width, top), (0, bottom, screen_width, screen_height))
        if area[2] > area[0] and area[3] > area[1]]
    return screen, empty_areas
