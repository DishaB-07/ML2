"""Downloads the YuNet face-detection model into the models/ folder.

Run once:  python download_model.py
Source: OpenCV Zoo (https://github.com/opencv/opencv_zoo), MIT license.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
# Expected SHA-256 (from the Git LFS pointer in the OpenCV Zoo repository)
EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + MODEL_NAME,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_NAME,
]

target = Path(__file__).resolve().parent / "models" / MODEL_NAME
target.parent.mkdir(exist_ok=True)


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if target.exists() and sha256_of(target) == EXPECTED_SHA256:
    print("Model already present and verified:", target)
    sys.exit(0)

for url in URLS:
    try:
        print("Downloading from", url)
        urllib.request.urlretrieve(url, target)
        if sha256_of(target) == EXPECTED_SHA256:
            print("Download OK and checksum verified:", target)
            sys.exit(0)
        print("Checksum mismatch, trying next source...")
    except Exception as error:
        print("Failed:", error)

target.unlink(missing_ok=True)
print("\nCould not download the model automatically.")
print("Download it manually from:")
print("  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet")
print("and put", MODEL_NAME, "inside the models/ folder.")
print("(main.py still works without it, using the Haar Cascade fallback.)")
sys.exit(1)
