"""Face detection with a pretrained model.
YuNet (OpenCV DNN) is used if its file exists, otherwise the Haar Cascade fallback."""
import sys

import cv2

import settings
from detections import Detection


class FaceDetector:
    def __init__(self):
        self.model = None
        self.name = ""
        self._load_yunet() or self._load_haar()
        if self.model is None:
            print("ERROR: no face model could be loaded. Check the models/ folder.")
            sys.exit(1)

    def _load_yunet(self):
        if not settings.YUNET_MODEL_FILE.exists():
            print("YuNet model not found (run: python download_model.py). Using Haar fallback.")
            return False
        if not hasattr(cv2, "FaceDetectorYN_create"):
            print("This OpenCV version has no YuNet support. Using Haar fallback.")
            return False
        try:
            self.model = cv2.FaceDetectorYN_create(
                str(settings.YUNET_MODEL_FILE), "", (settings.FACE_SEARCH_WIDTH, 360),
                settings.FACE_CONFIDENCE_LIMIT)
            self.name = "YuNet"
            return True
        except cv2.error as error:
            print("Could not load YuNet:", error)
            return False

    def _load_haar(self):
        if settings.HAAR_MODEL_FILE.exists():
            cascade = cv2.CascadeClassifier(str(settings.HAAR_MODEL_FILE))
            if not cascade.empty():
                self.model = cascade
                self.name = "Haar"

    def find_faces(self, frame):
        """Return a list of Detection boxes (one per face) in full-frame pixels."""
        frame_height, frame_width = frame.shape[:2]
        shrink = min(1.0, settings.FACE_SEARCH_WIDTH / frame_width)
        small_frame = frame
        if shrink < 1.0:
            small_frame = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)

        boxes = []   # (left, top, width, height, score)
        if self.name == "YuNet":
            self.model.setInputSize((small_frame.shape[1], small_frame.shape[0]))
            _, found = self.model.detect(small_frame)
            if found is not None:
                boxes = [(f[0], f[1], f[2], f[3], float(f[14])) for f in found]
        else:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            found = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            boxes = [(l, t, w, h, None) for l, t, w, h in found]

        faces = []
        for left, top, width, height, score in boxes:
            left, top = max(0, int(left / shrink)), max(0, int(top / shrink))
            width = min(int(width / shrink), frame_width - left)
            height = min(int(height / shrink), frame_height - top)
            faces.append(Detection(left, top, width, height, "Human Face", score))
        return faces
