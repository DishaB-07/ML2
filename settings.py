"""All settings in one place. Change a value here to tune the app."""
from pathlib import Path

# ---------- Folders and model files ----------
PROJECT_FOLDER = Path(__file__).resolve().parent
MODELS_FOLDER = PROJECT_FOLDER / "models"
CAPTURES_FOLDER = PROJECT_FOLDER / "captures"        # photos and videos are saved here
YUNET_MODEL_FILE = MODELS_FOLDER / "face_detection_yunet_2023mar.onnx"
HAAR_MODEL_FILE = MODELS_FOLDER / "haarcascade_frontalface_default.xml"

# ---------- Camera and window ----------
CAMERA_NUMBER = 0                  # try 1 or 2 if the default camera does not open
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
START_FULLSCREEN = True            # press F while running to switch
WINDOW_TITLE = "Human Face and Colour Detection"
RECORDING_FPS = 20.0

# ---------- Face detection ----------
FACE_CONFIDENCE_LIMIT = 0.7        # 0 to 1. Raise (e.g. 0.85) for fewer false faces
FACE_SEARCH_WIDTH = 640            # faces are searched on a frame this wide (faster)
FACE_EXTRA_MARGIN = 0.25           # colours are ignored on the face area plus this margin

# ---------- Colour detection ----------
COLOUR_SEARCH_WIDTH = 640          # colours are searched on a frame this wide (faster)
SMALLEST_REGION_FRACTION = 0.0015  # a colour region must cover at least this part of the frame
THINNEST_REGION_PIXELS = 10        # ignore very thin slivers
BLUR_AMOUNT = 5                    # blur before analysing colours (less camera noise)
CLEANUP_SIZE = 5                   # noise-removal size; also removes thin bands at object edges
MAX_BOXES = 30                     # draw at most this many colour boxes (largest first)
BLACK_BELOW_BRIGHTNESS = 45        # darker than this is Black
WHITE_ABOVE_BRIGHTNESS = 200       # brighter than this (and not colourful) is White

# Sensitivity 1-9: higher means paler colours also count as colours.
DEFAULT_SENSITIVITY = 5
MIN_SENSITIVITY = 1
MAX_SENSITIVITY = 9

# ---------- Box smoothing (stops flickering) ----------
KEEP_OLD_BOX_WEIGHT = 0.6          # 0 = jump to new box, 1 = never move
FRAMES_TO_CONFIRM = 2              # a box must be seen this many frames before it is drawn
FRAMES_TO_FORGET = 4               # a missing box stays this many frames before it is removed

# ---------- Colour names and the colours used to draw them (BGR) ----------
COLOUR_DRAW_COLOURS = {
    "Red": (60, 60, 240), "Orange": (0, 140, 255), "Yellow": (0, 230, 255),
    "Lime": (50, 255, 150), "Green": (60, 200, 60), "Dark Green": (40, 100, 20),
    "Teal": (128, 128, 0), "Cyan": (255, 255, 0), "Light Blue": (250, 200, 140),
    "Blue": (255, 100, 30), "Navy": (110, 30, 0), "Purple": (200, 60, 150),
    "Lavender": (250, 230, 230), "Magenta": (255, 0, 255), "Pink": (193, 150, 255),
    "Maroon": (40, 20, 128), "Brown": (30, 70, 140), "Beige": (190, 220, 245),
    "Olive": (0, 128, 128), "White": (255, 255, 255), "Gray": (150, 150, 150),
    "Black": (60, 60, 60),
}
COLOUR_NAMES = list(COLOUR_DRAW_COLOURS)
NEUTRAL_COLOURS = ("White", "Gray", "Black")
FACE_BOX_COLOUR = (80, 255, 90)    # green
