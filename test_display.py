"""
test_display.py

Student-facing display test:
- Captures live frames from the display port camera and USB/UVC camera
- Uses OpenCV to annotate the frame to simulate "robot perception"
- Displays both annotated camera panels on the 7" DSI touchscreen via RobotDisplay
- Uses on-screen buttons to toggle AUTO mode, STOP, SNAP, QUIT

Requirements:
- display.py (RobotDisplay) in same folder
- ui.py in same folder (spawned by RobotDisplay)
- picamera2, opencv-python, numpy installed
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from display import RobotDisplay

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except Exception:
    Picamera2 = None
    CAMERA_AVAILABLE = False


FRONT_CAMERA_ID = "front"
SIDE_CAMERA_ID = "side"
CAMERA_IDS = (FRONT_CAMERA_ID, SIDE_CAMERA_ID)
CAMERA_LABELS = {
    FRONT_CAMERA_ID: "FRONT CAMERA",
    SIDE_CAMERA_ID: "SIDE CAMERA",
}
CAMERA_COLOR_CONVERSIONS = {
    FRONT_CAMERA_ID: cv2.COLOR_BGRA2BGR,
    SIDE_CAMERA_ID: cv2.COLOR_RGBA2BGR,
}
CAMERA_TUNING_CONTROLS = {
    SIDE_CAMERA_ID: {
        "AeEnable": True,
        "ExposureValue": 1.4,
        "Brightness": 0.18,
        "Contrast": 1.25,
        "Sharpness": 2.0,
        "AfMode": 2,
    },
}
CAMERA_SOFTWARE_ENHANCEMENT = {
    SIDE_CAMERA_ID: {
        "alpha": 1.35,
        "beta": 34,
        "gamma": 0.72,
        "sharpen_amount": 0.55,
    },
}


# -------------------------
# OpenCV overlay helpers
# -------------------------
def draw_decision_banner(frame: np.ndarray, text: str, *, bg=(0, 255, 255), fg=(0, 0, 0)) -> None:
    """Draw a top banner with the robot's current decision."""
    _, w = frame.shape[:2]
    bar_h = 46
    cv2.rectangle(frame, (0, 0), (w, bar_h), bg, -1)
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fg, 2, cv2.LINE_AA)


def draw_roi(frame: np.ndarray, roi_xywh, *, label="ROI", color=(255, 0, 255)) -> None:
    """Draw a region-of-interest box."""
    if roi_xywh is None:
        return
    x, y, w2, h2 = map(int, roi_xywh)
    cv2.rectangle(frame, (x, y), (x + w2, y + h2), color, 2)
    cv2.putText(frame, label, (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, bbox_xywh, label: str, conf: float, *, color=(0, 255, 255)) -> None:
    """Draw a detection box + label + confidence."""
    if bbox_xywh is None:
        return
    x, y, bw, bh = map(int, bbox_xywh)

    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    text = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y0 = max(0, y - th - 10)
    cv2.rectangle(frame, (x, y0), (x + tw + 10, y), color, -1)

    cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_heading_arrow(frame: np.ndarray, steer: float) -> None:
    """
    Draw a heading arrow that indicates steering direction.
    steer: -1.0 (hard left) to +1.0 (hard right)
    """
    steer = float(max(-1.0, min(1.0, steer)))
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.82)
    length = int(min(w, h) * 0.25)

    ex = int(cx + steer * length)
    ey = cy - length

    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 0), 6, tipLength=0.25)


def draw_crosshair(frame: np.ndarray) -> None:
    """Draw a center crosshair."""
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)


def fit_frame_to_panel(frame_bgr: np.ndarray, width: int, height: int, *, bg=(0, 0, 0)) -> np.ndarray:
    """Fit a BGR frame into a fixed-size panel without cropping."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :] = bg
    if frame_bgr is None or frame_bgr.size == 0:
        return panel

    src_h, src_w = frame_bgr.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return panel

    scale = min(width / src_w, height / src_h)
    dw = max(1, int(round(src_w * scale)))
    dh = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
    dx = (width - dw) // 2
    dy = (height - dh) // 2
    panel[dy:dy + dh, dx:dx + dw, :] = resized
    return panel


def build_camera_placeholder(width: int, height: int, label: str, status: str) -> np.ndarray:
    """Build a fixed-size placeholder for an unavailable camera feed."""
    status = str(status)[:72]
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :] = (18, 18, 18)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (55, 55, 55), 2)
    cv2.putText(panel, label, (24, height // 2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(panel, status, (24, height // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 2, cv2.LINE_AA)
    return panel


def draw_camera_panel_label(canvas: np.ndarray, x: int, width: int, label: str, fps: float, status: str, available: bool) -> None:
    """Draw a compact camera label inside one panel of the combined display."""
    y0 = 48
    y1 = 78
    cv2.rectangle(canvas, (x + 8, y0), (x + width - 8, y1), (0, 0, 0), -1)
    text = f"{label} | {fps:4.1f} FPS" if available else f"{label} | {str(status)[:46]}"
    color = (255, 255, 255) if available else (120, 120, 120)
    cv2.putText(canvas, text, (x + 16, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_demo_annotations(frame: np.ndarray, *, auto_mode: bool, phase: float) -> None:
    """Draw the fake perception overlays used by the classroom display test."""
    h, w = frame.shape[:2]
    roi = (40, 86, 280, 204)
    x_center = int(w * (0.5 + 0.20 * np.sin(phase)))
    y_center = int(h * 0.42)
    bw, bh = 140, 140
    bbox = (x_center - bw // 2, y_center - bh // 2, bw, bh)
    steer = float(0.6 * np.sin(phase)) if auto_mode else 0.0

    draw_roi(frame, roi, label="Sign Search ROI")
    draw_detection(frame, bbox, "STOP", 0.92)
    draw_heading_arrow(frame, steer)
    cv2.putText(
        frame,
        "AUTO MODE" if auto_mode else "MANUAL MODE",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255) if auto_mode else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# -------------------------
# Camera helpers
# -------------------------
def select_picamera_numbers() -> Dict[str, Any]:
    """Select Picamera2 camera numbers for front and side roles."""
    if not CAMERA_AVAILABLE:
        return {FRONT_CAMERA_ID: None, SIDE_CAMERA_ID: None}

    try:
        camera_info = Picamera2.global_camera_info()
    except Exception as exc:
        print(f"Camera discovery failed; using default camera numbers: {exc}")
        return {FRONT_CAMERA_ID: 0, SIDE_CAMERA_ID: 1}

    for idx, info in enumerate(camera_info or []):
        print(f"Picamera2 camera {idx}: {info}")

    if not camera_info:
        return {FRONT_CAMERA_ID: 0, SIDE_CAMERA_ID: 1}

    usb_camera_numbers = []
    non_usb_camera_numbers = []
    for idx, info in enumerate(camera_info):
        if camera_info_looks_usb(info):
            usb_camera_numbers.append(idx)
        else:
            non_usb_camera_numbers.append(idx)

    front_num = non_usb_camera_numbers[0] if non_usb_camera_numbers else 0
    side_num = None
    for idx in usb_camera_numbers:
        if idx != front_num:
            side_num = idx
            break
    if side_num is None:
        for idx in range(len(camera_info)):
            if idx != front_num:
                side_num = idx
                break

    return {FRONT_CAMERA_ID: front_num, SIDE_CAMERA_ID: side_num}


def camera_info_looks_usb(info: Any) -> bool:
    """Return True when Picamera2 discovery metadata looks like a USB/UVC camera."""
    text = " ".join(str(value).lower() for value in info.values()) if isinstance(info, dict) else str(info).lower()
    return "usb" in text or "uvc" in text


def initialize_cameras(width: int, height: int):
    """Configure the front and side Picamera2 streams."""
    cameras = {camera_id: None for camera_id in CAMERA_IDS}
    camera_enabled = {camera_id: False for camera_id in CAMERA_IDS}
    camera_status = {camera_id: "not initialized" for camera_id in CAMERA_IDS}

    if not CAMERA_AVAILABLE:
        for camera_id in CAMERA_IDS:
            camera_status[camera_id] = "Picamera2 not available"
        return cameras, camera_enabled, camera_status

    camera_numbers = select_picamera_numbers()
    started_camera_ids = []

    for camera_id in CAMERA_IDS:
        camera_num = camera_numbers.get(camera_id)
        if camera_num is None:
            camera_status[camera_id] = "not detected"
            continue

        try:
            camera = Picamera2(camera_num)
            config = camera.create_video_configuration(
                main={"size": (width, height), "format": "XRGB8888"}
            )
            camera.configure(config)
            camera.start()
            cameras[camera_id] = camera
            camera_enabled[camera_id] = True
            camera_status[camera_id] = f"enabled on Picamera2({camera_num})"
            started_camera_ids.append(camera_id)
        except Exception as exc:
            cameras[camera_id] = None
            camera_enabled[camera_id] = False
            camera_status[camera_id] = f"init failed: {exc}"

    if started_camera_ids:
        time.sleep(1.5)

    for camera_id in started_camera_ids:
        camera = cameras.get(camera_id)
        if camera is None:
            continue
        try:
            meta = camera.capture_metadata()
            if "ColourGains" in meta:
                camera.set_controls({"AwbEnable": False, "ColourGains": meta["ColourGains"]})
        except Exception:
            pass
        apply_camera_tuning(camera_id, camera)

    return cameras, camera_enabled, camera_status


def apply_camera_tuning(camera_id: str, camera: Any) -> None:
    """Apply optional camera-specific image tuning controls."""
    requested_controls = CAMERA_TUNING_CONTROLS.get(camera_id, {})
    if not requested_controls:
        return

    supported_controls = getattr(camera, "camera_controls", {}) or {}
    applied: Dict[str, Any] = {}
    skipped = []

    for name, value in requested_controls.items():
        if supported_controls and name not in supported_controls:
            skipped.append(f"{name}=unsupported")
            continue
        control_info = supported_controls.get(name) if hasattr(supported_controls, "get") else None
        tuned_value = clamp_control_value(value, control_info)
        try:
            camera.set_controls({name: tuned_value})
            applied[name] = tuned_value
        except Exception as exc:
            skipped.append(f"{name}={exc}")

    if applied:
        print(f"{camera_id} camera tuning applied: {applied}")
    if skipped:
        print(f"{camera_id} camera tuning skipped: {', '.join(skipped)}")


def clamp_control_value(value: Any, control_info: Any) -> Any:
    """Clamp simple numeric Picamera2 controls when min/max metadata is available."""
    if not isinstance(value, (int, float)):
        return value
    if not isinstance(control_info, tuple) or len(control_info) < 2:
        return value

    low, high = control_info[0], control_info[1]
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return value

    tuned = max(low, min(high, value))
    if isinstance(value, int):
        return int(round(tuned))
    return float(tuned)


def camera_array_to_bgr(camera_id: str, frame_array: np.ndarray) -> np.ndarray:
    """Convert each Picamera2 output array to OpenCV-native BGR."""
    if frame_array.ndim == 3 and frame_array.shape[2] == 4:
        conversion = CAMERA_COLOR_CONVERSIONS.get(camera_id, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(frame_array, conversion)
    if frame_array.ndim == 3 and frame_array.shape[2] == 3:
        if camera_id == SIDE_CAMERA_ID:
            return cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        return frame_array
    if frame_array.ndim == 2:
        return cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported camera frame shape: {frame_array.shape}")


def enhance_camera_frame(camera_id: str, frame_bgr: np.ndarray) -> np.ndarray:
    """Apply optional per-camera software image enhancement after capture."""
    settings = CAMERA_SOFTWARE_ENHANCEMENT.get(camera_id)
    if not settings:
        return frame_bgr

    enhanced = frame_bgr
    gamma = float(settings.get("gamma", 1.0))
    if gamma > 0 and gamma != 1.0:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    alpha = float(settings.get("alpha", 1.0))
    beta = float(settings.get("beta", 0.0))
    if alpha != 1.0 or beta != 0.0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    sharpen_amount = float(settings.get("sharpen_amount", 0.0))
    if sharpen_amount > 0:
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        enhanced = cv2.addWeighted(enhanced, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

    return enhanced


def update_camera_frames(
    cameras: Dict[str, Any],
    camera_enabled: Dict[str, bool],
    camera_status: Dict[str, str],
    camera_frames: Dict[str, Any],
    camera_fps: Dict[str, float],
    last_frame_time: Dict[str, float],
    first_camera_frame_logged: Dict[str, bool],
    width: int,
    height: int,
    fps_alpha: float,
) -> None:
    """Capture the newest front and side camera frames."""
    now = time.time()
    for camera_id in CAMERA_IDS:
        camera = cameras.get(camera_id)
        if not camera_enabled.get(camera_id, False) or camera is None:
            camera_frames[camera_id] = None
            continue

        try:
            frame_array = camera.capture_array()
            frame_bgr = camera_array_to_bgr(camera_id, frame_array)
            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_bgr = enhance_camera_frame(camera_id, frame_bgr)
            camera_frames[camera_id] = frame_bgr
            camera_status[camera_id] = "live"

            if not first_camera_frame_logged[camera_id]:
                print(f"First {camera_id} camera frame captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
                first_camera_frame_logged[camera_id] = True

            dt = now - last_frame_time[camera_id]
            last_frame_time[camera_id] = now
            if dt > 0:
                fps_inst = 1.0 / dt
                current = camera_fps[camera_id]
                camera_fps[camera_id] = fps_inst if current == 0 else (fps_alpha * fps_inst + (1 - fps_alpha) * current)
        except Exception as exc:
            camera_frames[camera_id] = None
            camera_status[camera_id] = f"capture failed: {exc}"


def close_cameras(cameras: Dict[str, Any]) -> None:
    """Stop and close any camera objects that were opened."""
    for camera_id, camera in list(cameras.items()):
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.close()
        except Exception:
            pass
        cameras[camera_id] = None


# -------------------------
# Display composition
# -------------------------
def build_camera_panel(
    camera_id: str,
    camera_frames: Dict[str, Any],
    camera_status: Dict[str, str],
    width: int,
    height: int,
    *,
    auto_mode: bool,
    phase: float,
) -> np.ndarray:
    """Build one fixed-size display panel for a camera feed."""
    frame = camera_frames.get(camera_id)
    if frame is None:
        panel = build_camera_placeholder(
            width,
            height,
            CAMERA_LABELS[camera_id],
            camera_status.get(camera_id, "offline"),
        )
    else:
        panel = fit_frame_to_panel(frame.copy(), width, height)

    draw_crosshair(panel)
    if camera_id == FRONT_CAMERA_ID:
        draw_demo_annotations(panel, auto_mode=auto_mode, phase=phase)
    return panel


def compose_display_frame(
    camera_frames: Dict[str, Any],
    camera_status: Dict[str, str],
    camera_fps: Dict[str, float],
    width: int,
    height: int,
    *,
    auto_mode: bool,
    phase: float,
) -> np.ndarray:
    """Build the two-camera display canvas submitted to RobotDisplay."""
    panels = [
        build_camera_panel(camera_id, camera_frames, camera_status, width, height, auto_mode=auto_mode, phase=phase)
        for camera_id in CAMERA_IDS
    ]

    combined = np.hstack(panels)
    cv2.line(combined, (width, 0), (width, combined.shape[0]), (45, 45, 45), 2)
    draw_decision_banner(combined, "Action: BRAKE (STOP sign)" if auto_mode else "Action: MANUAL")

    for index, camera_id in enumerate(CAMERA_IDS):
        available = camera_frames.get(camera_id) is not None
        draw_camera_panel_label(
            combined,
            index * width,
            width,
            CAMERA_LABELS[camera_id],
            camera_fps.get(camera_id, 0.0),
            camera_status.get(camera_id, "offline"),
            available,
        )

    return combined


def build_status_line(auto_mode: bool, camera_frames: Dict[str, Any]) -> str:
    """Build the short status string shown in the display's top bar."""
    front_status = "OK" if camera_frames.get(FRONT_CAMERA_ID) is not None else "OFF"
    side_status = "OK" if camera_frames.get(SIDE_CAMERA_ID) is not None else "OFF"
    return f"Mode: {'AUTO' if auto_mode else 'MANUAL'} | Front: {front_status} | Side: {side_status}"


def set_display_status_if_changed(disp: RobotDisplay, text: str, last_status_text: str) -> str:
    """Avoid sending identical status updates every loop."""
    if text == last_status_text:
        return last_status_text
    disp.set_status(
        text,
        position="top",
        bg_color=(15, 15, 15),
        text_color=(255, 255, 255),
    )
    return text


# -------------------------
# Main test program
# -------------------------
def main():
    perception_width = 640
    perception_height = 480

    # --- Display/UI ---
    disp = RobotDisplay(fullscreen=True, fps=30, size=(800, 480), bg_color=(0, 0, 0))

    disp.set_buttons(
        [
            {"handle": "auto", "text": "AUTO", "bg_color": (30, 80, 200), "text_color": (255, 255, 255)},
            {"handle": "stop", "text": "STOP", "bg_color": (200, 60, 60), "text_color": (255, 255, 255)},
            {"handle": "snap", "text": "SNAP", "bg_color": (60, 160, 90), "text_color": (255, 255, 255)},
            {"handle": "quit", "text": "QUIT", "bg_color": (80, 80, 80), "text_color": (255, 255, 255)},
        ],
        position="bottom",
    )

    last_status_text = ""
    last_status_text = set_display_status_if_changed(
        disp,
        "Mode: MANUAL | Starting cameras...",
        last_status_text,
    )

    cameras = {camera_id: None for camera_id in CAMERA_IDS}
    camera_enabled = {camera_id: False for camera_id in CAMERA_IDS}
    camera_status = {camera_id: "not initialized" for camera_id in CAMERA_IDS}
    camera_frames = {camera_id: None for camera_id in CAMERA_IDS}
    camera_fps = {camera_id: 0.0 for camera_id in CAMERA_IDS}
    last_frame_time = {camera_id: time.time() for camera_id in CAMERA_IDS}
    first_camera_frame_logged = {camera_id: False for camera_id in CAMERA_IDS}

    auto_mode = False
    snap_count = 0
    phase = 0.0
    last_loop_time = time.time()
    fps_alpha = 0.15
    status_notice = ""
    status_notice_until = 0.0

    try:
        cameras, camera_enabled, camera_status = initialize_cameras(perception_width, perception_height)

        while True:
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now

            update_camera_frames(
                cameras,
                camera_enabled,
                camera_status,
                camera_frames,
                camera_fps,
                last_frame_time,
                first_camera_frame_logged,
                perception_width,
                perception_height,
                fps_alpha,
            )

            phase += dt * (1.5 if auto_mode else 0.7)

            display_frame = compose_display_frame(
                camera_frames,
                camera_status,
                camera_fps,
                perception_width,
                perception_height,
                auto_mode=auto_mode,
                phase=phase,
            )
            disp.set_frame(display_frame)

            if status_notice and now < status_notice_until:
                status_text = status_notice
            else:
                status_notice = ""
                status_text = build_status_line(auto_mode, camera_frames)
            last_status_text = set_display_status_if_changed(disp, status_text, last_status_text)

            for ev in disp.poll_events():
                if ev == "auto":
                    auto_mode = not auto_mode

                elif ev == "stop":
                    auto_mode = False
                    status_notice = "Mode: MANUAL | Stopped"
                    status_notice_until = time.time() + 1.5

                elif ev == "snap":
                    snap_count += 1
                    saved_files = []
                    for camera_id in CAMERA_IDS:
                        frame = camera_frames.get(camera_id)
                        if frame is None:
                            continue
                        fname = f"snapshot_{snap_count:03d}_{camera_id}.jpg"
                        try:
                            cv2.imwrite(fname, frame)
                            saved_files.append(fname)
                        except Exception as exc:
                            status_notice = f"Snapshot failed: {exc}"
                            status_notice_until = time.time() + 2.5
                            break
                    if saved_files:
                        status_notice = f"Saved {', '.join(saved_files)}"
                    elif not status_notice:
                        status_notice = "Snapshot skipped: no camera frames available"
                    status_notice_until = time.time() + 2.5

                elif ev == "quit":
                    return

            time.sleep(0.001)

    finally:
        disp.close()
        close_cameras(cameras)


if __name__ == "__main__":
    main()
