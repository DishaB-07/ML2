# Human Face and Colour Detection System

A real-time, **full-screen** computer vision app (Python + OpenCV). It detects **human faces** and **22 named colours** at the same time, from your **live webcam** or from an **uploaded image**. Switch between the two while the app is running.

Colours: Red, Orange, Yellow, Lime, Green, Dark Green, Teal, Cyan, Light Blue, Blue, Navy, Purple, Lavender, Magenta, Pink, Maroon, Brown, Beige, Olive, White, Gray, Black.

## Folder structure
```
human_colour_detection/
├── main.py              # the app: window, keys, main loop
├── settings.py          # ALL settings in one place (camera, thresholds, colours)
├── face_detector.py     # face detection (YuNet model, Haar fallback)
├── colour_detector.py   # HSV colour naming + masks + contours
├── detections.py        # Detection box + BoxSmoother (stops flicker)
├── drawing.py           # boxes, labels, bars, legend
├── input_sources.py     # webcam and image-file loading
├── app_state.py         # what changes while running (modes, toggles)
├── download_model.py    # downloads the YuNet face model
├── requirements.txt
├── models/
│   └── haarcascade_frontalface_default.xml   # bundled fallback model
└── tests/
    └── test_detection.py                     # automatic checks (no webcam needed)
```
`models/face_detection_yunet_2023mar.onnx` is added by `download_model.py` (step 4). The `captures/` folder is created automatically for photos and videos.

## Face model used
- **Preferred: YuNet** - a small pretrained CNN face detector from the OpenCV Zoo (MIT licence, about 230 KB). Runs on a normal CPU through `cv2.FaceDetectorYN`. More robust than Haar for tilted faces and different lighting. It also gives a confidence score, shown next to `Human Face`.
- **Fallback: Haar Cascade** - OpenCV's classic pretrained detector, bundled in `models/`. Used automatically if the YuNet file is missing. The top bar shows which one is running.
- This is a *face* detector, not a whole-body "person" detector like generic YOLO.

## Setup on Windows (VS Code)
1. Install Python 3.9+ from python.org (tick **Add Python to PATH**).
2. Extract the ZIP and open the `human_colour_detection` folder in VS Code (File > Open Folder).
3. Open the terminal (Ctrl + `) and run:
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
   If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
4. Download the YuNet model (internet needed, once):
```
python download_model.py
```
5. Run:
```
python main.py
```
Other ways to start:
```
python main.py --image "C:\path\to\picture.jpg"     (start with an image)
python main.py --camera 1                           (use another camera)
```
6. In VS Code: Ctrl+Shift+P > **Python: Select Interpreter** > choose `.venv`.

## Controls
| Key | What it does |
|---|---|
| `1` | **Option 1 - single colour**: detect one colour at a time |
| `N` / `P` | next / previous colour (single mode, all 22) |
| `R` `G` `B` `Y` | jump to Red, Green, Blue, Yellow (single mode) |
| `2` | **Option 2 - all colours** (default): find and name every colour |
| `U` | **upload an image** (opens a file window) and detect on it |
| `C` | go back to the **live camera** |
| `E` | hide/show White, Gray, Black (walls and shadows) |
| `+` / `-` | colour sensitivity 1-9 (higher counts paler colours too) |
| `W` | white balance on/off (see note below) |
| `S` | save a photo to `captures/` |
| `V` | start/stop video recording to `captures/` |
| `F` | full screen on/off |
| `M` | show/hide the black-and-white colour mask window |
| `Q` or `Esc` | quit |

Left-click anywhere to read the HSV value and colour name of that pixel. If no webcam is found, the app still starts so you can use `U` to test with images.

**White balance (`W`)** assumes the scene averages to gray. It helps under yellow lamp light but can tint things wrongly when one big colour fills the screen. Leave it off unless colours look shifted.

## What was improved for precision
- Frame is blurred first to reduce camera noise.
- Every pixel is classified into one of 22 colours using a pre-computed lookup table (fast, and single and all modes use the same rules).
- Morphological opening/closing cleans each mask; the 5x5 opening also removes false colour bands along object edges.
- Tiny and sliver-shaped regions are ignored.
- The face area is ignored, so skin and hair are not boxed as Orange/Brown/Beige.
- A box must be seen on 2 frames before it is drawn and is blended with the previous frame, so boxes do not flicker.
- The legend shows how much of the frame each colour covers.
Colour detection still depends on lighting, camera quality and object size. Click an object to read its HSV, then adjust `name_the_colour()` in `colour_detector.py` if your camera shifts colours.

## Settings (`settings.py`)
`CAMERA_NUMBER`, `CAMERA_WIDTH/HEIGHT`, `START_FULLSCREEN`, `FACE_CONFIDENCE_LIMIT`, `SMALLEST_REGION_FRACTION`, `BLUR_AMOUNT`, `CLEANUP_SIZE`, `DEFAULT_SENSITIVITY`, smoothing values, and the colour names/draw colours.

## Run the automatic tests
```
python -m unittest discover -s tests -v
```

## Troubleshooting
- **Cannot open camera**: close Zoom/Teams/Camera app; Windows Settings > Privacy > Camera > allow desktop apps; try `python main.py --camera 1`.
- **Top bar says Haar**: YuNet file missing. Run `python download_model.py`, or download `face_detection_yunet_2023mar.onnx` from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet into `models/`.
- **`cv2` not found**: the virtual environment is not activated or the interpreter is not selected.
- **Upload window does not appear**: it may be behind the full-screen window; press `F` to leave full screen and try `U` again. Or start with `--image`.
- **A colour is misnamed or missed**: press `M` to see the mask, click the object to read its HSV, change `+`/`-` sensitivity, or edit `name_the_colour()`.
- **Too many boxes**: press `E`, lower the sensitivity with `-`, or raise `SMALLEST_REGION_FRACTION`.
- **Low FPS**: lower `CAMERA_WIDTH`/`CAMERA_HEIGHT` (for example 640 x 480).

ed smoothing? Single-frame detections flicker; averaging gives stable boxes.
