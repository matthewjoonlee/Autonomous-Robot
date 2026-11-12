"""Drawing instructions from perception modules, drawn by the main controller.

Week 10/11 design: workers return small descriptions, not annotated images; the
controller draws them on the latest display frame. Each annotation is a dict:

  {"kind": "rect"|"line"|"point"|"text", "camera": "front"|"side",
   "xy": (x, y), "xy2": (x2, y2),      # rect corners / line ends
   "relative": True,                    # coordinates as fractions of width/height
   "text": "...", "label": "...", "color": (b, g, r), "thickness": 2}
"""

import cv2


def _point(frame, xy, relative):
    if relative:
        height, width = frame.shape[:2]
        return int(round(xy[0] * (width - 1))), int(round(xy[1] * (height - 1)))
    return int(round(xy[0])), int(round(xy[1]))


def draw(frame, annotations, camera):
    """Draw every annotation for `camera` onto frame (in place); returns frame."""
    for item in annotations or ():
        if item.get("camera", "front") != camera:
            continue
        color = tuple(int(c) for c in item.get("color", (0, 255, 255)))
        thickness = int(item.get("thickness", 2))
        relative = bool(item.get("relative", False))
        kind = item.get("kind")
        if kind == "rect":
            a, b = _point(frame, item["xy"], relative), _point(frame, item["xy2"], relative)
            cv2.rectangle(frame, a, b, color, thickness)
            if item.get("label"):
                label(frame, item["label"], (a[0] + 4, max(16, a[1] + 16)), color, scale=0.5)
        elif kind == "line":
            cv2.line(frame, _point(frame, item["xy"], relative), _point(frame, item["xy2"], relative),
                     color, thickness)
        elif kind == "point":
            cv2.circle(frame, _point(frame, item["xy"], relative), int(item.get("radius", 5)), color, -1)
        elif kind == "text":
            label(frame, item["text"], _point(frame, item["xy"], relative), color,
                  scale=float(item.get("scale", 0.7)))
    return frame


def label(frame, text, origin, color, scale=0.7):
    """Readable text: dark outline under colored text."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
