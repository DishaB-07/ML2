"""Human Face and Colour Detection System - real-time, full-screen, webcam or uploaded image.

Run:  python main.py
      python main.py --image path\\to\\picture.jpg      (start with an image)
      python main.py --camera 1                         (use another camera)

Keys (also shown on screen):
  1 Single colour   2 All colours   N / P next / previous colour   R G B Y quick colours
  E show/hide White-Gray-Black      U upload an image               C back to live camera
  W white balance   + / - sensitivity   S save photo   V start/stop recording
  F full screen     M mask window       Q or Esc quit
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    print("OpenCV is missing. Run: python -m pip install -r requirements.txt")
    sys.exit(1)

import settings
from app_state import AppState
from colour_detector import (build_colour_lookup, face_area_to_ignore, find_colour_regions,
                             name_the_colour, saturation_limit_for)
from detections import BoxSmoother
from drawing import (draw_colour_boxes, draw_colour_legend, draw_face_boxes,
                     draw_overlay, make_waiting_screen)
from face_detector import FaceDetector
from input_sources import (ask_for_image_file, fit_image_to_screen,
                           open_camera, read_image_file)

MASK_WINDOW_TITLE = "Colour mask (M to hide)"


class DetectionApp:
    def __init__(self, camera_number, image_path=None):
        self.state = AppState()
        self.face_detector = FaceDetector()
        print("Face detector in use:", self.face_detector.name)

        self.colour_lookup = build_colour_lookup(self.state.sensitivity)
        self.face_smoother = BoxSmoother()
        self.colour_smoother = BoxSmoother()
        self.waiting_screen = make_waiting_screen(settings.CAMERA_WIDTH, settings.CAMERA_HEIGHT)

        self.clean_frame = None        # the frame before anything is drawn on it
        self.mask_image = None
        self.video_writer = None
        self.photo_requested = False
        self.recording_requested = False

        self.camera = open_camera(camera_number)
        if self.camera is None:
            print(f"WARNING: cannot open camera {camera_number}. Close other apps using it, check the "
                  "Windows camera privacy settings, or run with  --camera 1 .\n"
                  "You can still test with an image: press U in the app or use  --image path.")
            self.state.input_source = "image"
            self.show_notice("No camera found - press U to upload an image", 6)
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)

        if image_path:
            self.load_image(image_path)

    # ---------------- small helpers ----------------
    def show_notice(self, text, seconds=2.5):
        self.state.notice_text = text
        self.state.notice_until = time.time() + seconds

    def reset_boxes(self):
        self.face_smoother.reset()
        self.colour_smoother.reset()

    def apply_fullscreen(self):
        flag = cv2.WINDOW_FULLSCREEN if self.state.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(settings.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, flag)

    # ---------------- input: camera or image ----------------
    def load_image(self, path):
        image = read_image_file(path)
        if image is None:
            print("Could not read the image:", path)
            self.show_notice("Could not read that image")
            return
        picture, empty_areas = fit_image_to_screen(image, settings.CAMERA_WIDTH, settings.CAMERA_HEIGHT)
        self.state.uploaded_picture = picture
        self.state.empty_areas = empty_areas
        self.state.uploaded_name = Path(path).name
        self.state.input_source = "image"
        self.reset_boxes()
        self.show_notice("Image loaded")

    def upload_image(self):
        path = ask_for_image_file()
        if path:
            self.load_image(path)

    def use_camera(self):
        if self.camera is None:
            self.show_notice("No camera available")
            return
        self.state.input_source = "camera"
        self.reset_boxes()
        self.show_notice("Live camera")

    def get_next_frame(self):
        """Return the next picture to analyse, or None if the camera stopped working."""
        if self.state.input_source == "camera":
            success, frame = self.camera.read()
            if not success:
                print("ERROR: failed to read a frame from the camera.")
                return None
            return cv2.flip(frame, 1)      # mirror view feels natural
        picture = self.state.uploaded_picture
        return (picture if picture is not None else self.waiting_screen).copy()

    # ---------------- detection + drawing ----------------
    def detect_and_draw(self, frame):
        """Run face + colour detection on the frame and draw the results. Returns the face count."""
        state = self.state
        scale = frame.shape[0] / 720.0
        has_picture = state.input_source == "camera" or state.uploaded_picture is not None
        if not has_picture:
            self.mask_image = None
            return 0

        faces = self.face_smoother.update(self.face_detector.find_faces(frame))

        if state.colour_mode == "all":
            colours_to_find, show_neutrals = None, state.show_neutrals
        else:
            colours_to_find, show_neutrals = [state.single_colour], True
        areas_to_ignore = [face_area_to_ignore(face) for face in faces]
        if state.input_source == "image":
            areas_to_ignore += state.empty_areas      # do not treat the dark borders as a colour
        regions, self.mask_image, colour_shares = find_colour_regions(
            frame, self.colour_lookup, areas_to_ignore, colours_to_find, show_neutrals, state.use_white_balance)
        regions = self.colour_smoother.update(regions)

        draw_colour_boxes(frame, regions, scale)
        draw_face_boxes(frame, faces, scale)
        draw_colour_legend(frame, regions, colour_shares, scale)
        return len(faces)

    # ---------------- photo and video ----------------
    def save_photo(self, frame):
        settings.CAPTURES_FOLDER.mkdir(exist_ok=True)
        path = settings.CAPTURES_FOLDER / f"photo_{datetime.now():%Y%m%d_%H%M%S}.png"
        success, encoded = cv2.imencode(".png", frame)
        if success:
            encoded.tofile(str(path))
            self.show_notice(f"Photo saved: captures/{path.name}")
            print("Saved", path)
        else:
            self.show_notice("Could not save the photo")

    def toggle_recording(self, frame):
        if self.state.is_recording:
            self.stop_recording()
            self.show_notice("Recording saved in the captures folder")
            return
        settings.CAPTURES_FOLDER.mkdir(exist_ok=True)
        path = settings.CAPTURES_FOLDER / f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), settings.RECORDING_FPS, (width, height))
        if writer.isOpened():
            self.video_writer = writer
            self.state.is_recording = True
            self.show_notice("Recording started")
        else:
            self.show_notice("Could not start recording")

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        self.state.is_recording = False

    # ---------------- mouse and keyboard ----------------
    def on_mouse(self, event, x, y, flags, parameter):
        """Left-click shows the HSV value and colour name of the clicked pixel."""
        frame = self.clean_frame
        if event != cv2.EVENT_LBUTTONDOWN or frame is None:
            return
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return
        patch = frame[max(0, y - 2):y + 3, max(0, x - 2):x + 3]       # average a small patch
        average_colour = patch.reshape(-1, 3).mean(axis=0).astype("uint8").reshape(1, 1, 3)
        hue, saturation, brightness = (int(n) for n in cv2.cvtColor(average_colour, cv2.COLOR_BGR2HSV)[0, 0])
        name = name_the_colour(hue, saturation, brightness, saturation_limit_for(self.state.sensitivity))
        self.state.probe_text = f"Clicked HSV = ({hue}, {saturation}, {brightness})  ->  {name}"
        print(self.state.probe_text)

    def change_sensitivity(self, step):
        state = self.state
        new_level = min(settings.MAX_SENSITIVITY, max(settings.MIN_SENSITIVITY, state.sensitivity + step))
        if new_level != state.sensitivity:
            state.sensitivity = new_level
            self.colour_lookup = build_colour_lookup(new_level)
            self.reset_boxes()

    def handle_key(self, key):
        """React to a key press. Returns False when the user wants to quit."""
        state = self.state
        if key in (ord("q"), ord("Q"), 27):          # 27 is the Esc key
            return False
        if 65 <= key <= 90:                           # treat capital letters as small letters
            key += 32

        if key == ord("1"):
            state.colour_mode = "single"
            self.reset_boxes()
        elif key == ord("2"):
            state.colour_mode = "all"
            self.reset_boxes()
        elif key == ord("e"):
            state.show_neutrals = not state.show_neutrals
            self.reset_boxes()
        elif key == ord("m"):
            state.show_mask = not state.show_mask
            if not state.show_mask:
                cv2.destroyWindow(MASK_WINDOW_TITLE)
        elif key == ord("f"):
            state.fullscreen = not state.fullscreen
            self.apply_fullscreen()
        elif key == ord("u"):
            self.upload_image()
        elif key == ord("c"):
            self.use_camera()
        elif key == ord("w"):
            state.use_white_balance = not state.use_white_balance
            self.reset_boxes()
        elif key in (ord("+"), ord("=")):
            self.change_sensitivity(+1)
        elif key in (ord("-"), ord("_")):
            self.change_sensitivity(-1)
        elif key == ord("s"):
            self.photo_requested = True
        elif key == ord("v"):
            self.recording_requested = True
        elif state.colour_mode == "single":
            colour_names = settings.COLOUR_NAMES
            position = colour_names.index(state.single_colour)
            quick_colours = {ord("r"): "Red", ord("g"): "Green", ord("b"): "Blue", ord("y"): "Yellow"}
            if key == ord("n"):
                state.single_colour = colour_names[(position + 1) % len(colour_names)]
                self.reset_boxes()
            elif key == ord("p"):
                state.single_colour = colour_names[(position - 1) % len(colour_names)]
                self.reset_boxes()
            elif key in quick_colours:
                state.single_colour = quick_colours[key]
                self.reset_boxes()
        return True

    # ---------------- main loop ----------------
    def run(self):
        cv2.namedWindow(settings.WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(settings.WINDOW_TITLE, self.on_mouse)
        self.apply_fullscreen()

        frames_per_second, last_time = 0.0, time.time()
        try:
            while True:
                frame = self.get_next_frame()
                if frame is None:
                    break
                self.clean_frame = frame.copy()

                face_count = self.detect_and_draw(frame)

                now = time.time()
                current_fps = 1.0 / max(now - last_time, 1e-6)
                frames_per_second = 0.9 * frames_per_second + 0.1 * current_fps if frames_per_second else 30.0
                last_time = now
                draw_overlay(frame, self.state, face_count, self.face_detector.name, frames_per_second, now)

                if self.photo_requested:
                    self.photo_requested = False
                    self.save_photo(frame)
                if self.recording_requested:
                    self.recording_requested = False
                    self.toggle_recording(frame)
                if self.state.is_recording:
                    self.video_writer.write(frame)

                if self.state.show_mask and self.mask_image is not None:
                    cv2.imshow(MASK_WINDOW_TITLE, self.mask_image)
                cv2.imshow(settings.WINDOW_TITLE, frame)

                wait_time = 1 if self.state.input_source == "camera" else 30   # images need less CPU
                if not self.handle_key(cv2.waitKey(wait_time) & 0xFF):
                    break
        finally:
            self.stop_recording()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()


def read_arguments():
    parser = argparse.ArgumentParser(description="Human Face and Colour Detection")
    parser.add_argument("--camera", type=int, default=settings.CAMERA_NUMBER, help="camera number (default 0)")
    parser.add_argument("--image", help="start with this image instead of the live camera")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = read_arguments()
    DetectionApp(arguments.camera, arguments.image).run()
