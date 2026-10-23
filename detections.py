"""A simple 'Detection' (one box on screen) and a smoother that stops boxes flickering."""
from dataclasses import dataclass
from typing import Optional

import settings


@dataclass
class Detection:
    """One detected face or colour region."""
    left: int
    top: int
    width: int
    height: int
    label: str                      # "Human Face" or a colour name
    score: Optional[float] = None   # face confidence (0-1) when the model gives one

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height


def overlap_ratio(box_a, box_b):
    """How much two boxes overlap, from 0 (none) to 1 (identical)."""
    overlap_width = min(box_a.right, box_b.right) - max(box_a.left, box_b.left)
    overlap_height = min(box_a.bottom, box_b.bottom) - max(box_a.top, box_b.top)
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    overlap_area = overlap_width * overlap_height
    total_area = box_a.width * box_a.height + box_b.width * box_b.height - overlap_area
    return overlap_area / total_area


class BoxSmoother:
    """Keeps boxes steady between video frames.
    - A new box is drawn only after it was seen a few frames in a row.
    - A box that disappears for a moment stays for a few frames.
    - Box positions are blended with the previous frame so they do not shake."""

    def __init__(self):
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, new_detections):
        for track in self.tracks:
            track["found_this_frame"] = False

        for detection in new_detections:
            best_track, best_overlap = None, 0.2
            for track in self.tracks:
                if track["found_this_frame"] or track["box"].label != detection.label:
                    continue
                overlap = overlap_ratio(track["box"], detection)
                if overlap > best_overlap:
                    best_track, best_overlap = track, overlap

            if best_track is None:
                self.tracks.append({"box": detection, "times_seen": 1,
                                    "frames_missing": 0, "found_this_frame": True})
            else:
                old = best_track["box"]
                keep = settings.KEEP_OLD_BOX_WEIGHT
                blend = lambda old_value, new_value: int(old_value * keep + new_value * (1 - keep))
                best_track["box"] = Detection(
                    blend(old.left, detection.left), blend(old.top, detection.top),
                    blend(old.width, detection.width), blend(old.height, detection.height),
                    detection.label, detection.score)
                best_track["times_seen"] += 1
                best_track["frames_missing"] = 0
                best_track["found_this_frame"] = True

        for track in self.tracks:
            if not track["found_this_frame"]:
                track["frames_missing"] += 1
        self.tracks = [t for t in self.tracks if t["frames_missing"] <= settings.FRAMES_TO_FORGET]

        return [t["box"] for t in self.tracks if t["times_seen"] >= settings.FRAMES_TO_CONFIRM]
