"""Everything that changes while the app is running (modes, toggles, messages)."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import settings


@dataclass
class AppState:
    input_source: str = "camera"            # "camera" (live) or "image" (uploaded picture)
    colour_mode: str = "all"                # "single" (option 1) or "all" (option 2)
    single_colour: str = "Red"              # colour used in single mode
    show_neutrals: bool = True              # show White / Gray / Black boxes
    sensitivity: int = settings.DEFAULT_SENSITIVITY
    use_white_balance: bool = False
    show_mask: bool = False
    fullscreen: bool = settings.START_FULLSCREEN
    is_recording: bool = False
    uploaded_picture: Optional[np.ndarray] = None   # uploaded image, fitted to the screen
    uploaded_name: str = ""
    empty_areas: List[Tuple[int, int, int, int]] = field(default_factory=list)  # dark borders around the image
    notice_text: str = ""                   # short message shown on screen
    notice_until: float = 0.0               # time when the message disappears
    probe_text: str = "Click anywhere to read the HSV value and colour name"
