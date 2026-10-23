"""Everything that draws on the screen: boxes, labels, bars and the colour legend."""
import cv2
import numpy as np

import settings

FONT = cv2.FONT_HERSHEY_DUPLEX


def how_bright(colour):
    """Rough brightness (0-255) of a BGR colour, used to choose black or white text."""
    blue, green, red = colour
    return 0.114 * blue + 0.587 * green + 0.299 * red


def darken_area(frame, left, top, right, bottom, strength=0.55):
    """Draw a see-through dark rectangle."""
    left, top = max(0, left), max(0, top)
    right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
    if right > left and bottom > top:
        area = frame[top:bottom, left:right]
        cv2.addWeighted(np.zeros_like(area), strength, area, 1 - strength, 0, dst=area)


def draw_pill(frame, text, left, top, fill_colour, scale, text_colour=None, text_size=0.55):
    """Draw a rounded label with text. Returns the x position of its right edge."""
    text_size *= scale
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, text_size, 1)
    padding = int(9 * scale)
    right, bottom = int(left + text_width + 2 * padding), int(top + text_height + 2 * padding)
    left, top = int(left), int(top)
    radius = (bottom - top) // 2
    cv2.rectangle(frame, (left + radius, top), (right - radius, bottom), fill_colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (left + radius, top + radius), radius, fill_colour, -1, cv2.LINE_AA)
    cv2.circle(frame, (right - radius, top + radius), radius, fill_colour, -1, cv2.LINE_AA)
    if text_colour is None:
        text_colour = (0, 0, 0) if how_bright(fill_colour) > 120 else (255, 255, 255)
    cv2.putText(frame, text, (left + padding, bottom - padding), FONT, text_size, text_colour, 1, cv2.LINE_AA)
    return right


def draw_corner_box(frame, box, colour, scale, thickness=3):
    """Thin box with thick corner marks. A contrasting outline keeps it visible on any background."""
    outline_colour = (0, 0, 0) if how_bright(colour) > 110 else (255, 255, 255)
    thickness = max(2, int(thickness * scale))
    corner_length = max(12, int(min(box.width, box.height) * 0.22))
    cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), colour, 1, cv2.LINE_AA)
    corners = [(box.left, box.top, 1, 1), (box.right, box.top, -1, 1),
               (box.left, box.bottom, 1, -1), (box.right, box.bottom, -1, -1)]
    for line_width, line_colour in ((thickness + 3, outline_colour), (thickness, colour)):
        for corner_x, corner_y, step_x, step_y in corners:
            cv2.line(frame, (corner_x, corner_y), (corner_x + step_x * corner_length, corner_y),
                     line_colour, line_width, cv2.LINE_AA)
            cv2.line(frame, (corner_x, corner_y), (corner_x, corner_y + step_y * corner_length),
                     line_colour, line_width, cv2.LINE_AA)


def draw_label_for_box(frame, text, box, colour, scale):
    """Put a label above the box (or below it if there is no space above)."""
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.55 * scale, 1)
    label_height = text_height + int(18 * scale)
    label_top = box.top - label_height - 4
    if label_top < 0:
        label_top = box.bottom + 4
    label_left = min(max(0, box.left), max(0, frame.shape[1] - text_width - int(30 * scale)))
    draw_pill(frame, text, label_left, label_top, colour, scale)


def draw_face_boxes(frame, faces, scale):
    for face in faces:
        draw_corner_box(frame, face, settings.FACE_BOX_COLOUR, scale, thickness=4)
        text = face.label if face.score is None else f"{face.label}  {int(face.score * 100)}%"
        draw_label_for_box(frame, text, face, settings.FACE_BOX_COLOUR, scale)


def draw_colour_boxes(frame, colour_regions, scale):
    for region in colour_regions:
        colour = settings.COLOUR_DRAW_COLOURS[region.label]
        draw_corner_box(frame, region, colour, scale)
        draw_label_for_box(frame, region.label, region, colour, scale)


def draw_colour_legend(frame, colour_regions, colour_shares, scale):
    """Panel on the left listing the colours found and how much of the frame each covers."""
    names = []
    for region in colour_regions:
        if region.label not in names:
            names.append(region.label)
    names = names[:10]
    if not names:
        return
    row_height = int(30 * scale)
    left, top = int(16 * scale), int(158 * scale)
    darken_area(frame, left, top, left + int(215 * scale), top + row_height * (len(names) + 1) + int(8 * scale))
    cv2.putText(frame, "DETECTED COLOURS", (left + int(12 * scale), top + int(22 * scale)),
                FONT, 0.45 * scale, (220, 220, 220), 1, cv2.LINE_AA)
    swatch = int(16 * scale)
    for row, name in enumerate(names):
        bottom = top + row_height * (row + 1) + int(24 * scale)
        swatch_left = left + int(12 * scale)
        cv2.rectangle(frame, (swatch_left, bottom - swatch), (swatch_left + swatch, bottom),
                      settings.COLOUR_DRAW_COLOURS[name], -1)
        cv2.rectangle(frame, (swatch_left, bottom - swatch), (swatch_left + swatch, bottom), (255, 255, 255), 1)
        share_text = f"{colour_shares[name]:.0f}%" if name in colour_shares else ""
        cv2.putText(frame, f"{name}  {share_text}", (left + int(40 * scale), bottom - 1),
                    FONT, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame, state, face_count, detector_name, fps, current_time):
    """Top bar, status labels, bottom help bar and the short notice message."""
    height, width = frame.shape[:2]
    scale = height / 720.0   # everything grows or shrinks with the frame size

    # Top bar: title and FPS
    darken_area(frame, 0, 0, width, int(62 * scale), 0.6)
    cv2.putText(frame, "FACE + COLOUR DETECTION", (int(20 * scale), int(40 * scale)),
                FONT, 0.85 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    fps_text = f"{fps:4.0f} FPS"
    (text_width, _), _ = cv2.getTextSize(fps_text, FONT, 0.7 * scale, 1)
    cv2.putText(frame, fps_text, (width - text_width - int(20 * scale), int(38 * scale)),
                FONT, 0.7 * scale, (120, 255, 160), 1, cv2.LINE_AA)

    gap = int(10 * scale)
    grey, light_text = (70, 70, 70), (230, 230, 230)

    # Row 1: faces, colour mode, input source
    left, top = int(16 * scale), int(74 * scale)
    left = draw_pill(frame, f"FACES  {face_count}", left, top, settings.FACE_BOX_COLOUR, scale) + gap
    if state.colour_mode == "all":
        left = draw_pill(frame, "MODE 2: ALL COLOURS", left, top, (255, 170, 60), scale) + gap
    else:
        left = draw_pill(frame, f"MODE 1: SINGLE  -  {state.single_colour}", left, top,
                         settings.COLOUR_DRAW_COLOURS[state.single_colour], scale) + gap
    if state.input_source == "camera":
        draw_pill(frame, "LIVE CAMERA", left, top, (230, 160, 90), scale)
    else:
        draw_pill(frame, f"IMAGE: {state.uploaded_name[:28]}", left, top, (90, 170, 230), scale)

    # Row 2: small status labels
    left, top = int(16 * scale), int(112 * scale)
    left = draw_pill(frame, detector_name, left, top, grey, scale, light_text, 0.45) + gap
    left = draw_pill(frame, f"SENSITIVITY {state.sensitivity}", left, top, grey, scale, light_text, 0.45) + gap
    if state.use_white_balance:
        left = draw_pill(frame, "WHITE BALANCE ON", left, top, grey, scale, light_text, 0.45) + gap
    if state.colour_mode == "all" and not state.show_neutrals:
        left = draw_pill(frame, "NEUTRALS OFF", left, top, grey, scale, light_text, 0.45) + gap
    if state.is_recording:
        draw_pill(frame, "REC", left, top, (40, 40, 220), scale, (255, 255, 255), 0.45)

    # Bottom bar: key help and colour probe text
    darken_area(frame, 0, height - int(84 * scale), width, height, 0.6)
    help_lines = ["1 Single colour   2 All colours   N/P Next/Prev colour   R G B Y Quick colours   E Neutrals",
                  "U Upload image   C Camera   W White balance   +/- Sensitivity   S Photo   V Record   F Fullscreen   M Mask   Q Quit"]
    for line_number, line in enumerate(help_lines):
        cv2.putText(frame, line, (int(20 * scale), height - int((60 - line_number * 22) * scale)),
                    FONT, 0.48 * scale, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, state.probe_text, (int(20 * scale), height - int(12 * scale)),
                FONT, 0.5 * scale, (0, 230, 255), 1, cv2.LINE_AA)

    # Short message (for example "Photo saved")
    if state.notice_text and current_time < state.notice_until:
        (text_width, _), _ = cv2.getTextSize(state.notice_text, FONT, 0.6 * scale, 1)
        draw_pill(frame, state.notice_text, (width - text_width) // 2 - int(15 * scale),
                  height - int(150 * scale), (255, 255, 255), scale, (0, 0, 0), 0.6)


def make_waiting_screen(width, height):
    """Screen shown when there is no camera and no image yet."""
    screen = np.full((height, width, 3), (35, 30, 28), np.uint8)
    scale = height / 720.0
    lines = [("No camera image yet", 1.2, (255, 255, 255)),
             ("Press U to upload an image", 0.8, (0, 230, 255)),
             ("Press C to use the live camera", 0.8, (230, 230, 230))]
    y = int(height * 0.42)
    for text, size, colour in lines:
        (text_width, _), _ = cv2.getTextSize(text, FONT, size * scale, 1)
        cv2.putText(screen, text, ((width - text_width) // 2, y), FONT, size * scale, colour, 1, cv2.LINE_AA)
        y += int(60 * scale)
    return screen
