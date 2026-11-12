"""Shared camera setup: open the front and side cameras at their full field of view.

Measured with camera_fov_check.py on 2026-11-12 (fov_check/session_20261112_152328, _152614):
- Front, Camera Module 3 Wide (imx708). A 640x480 request made Picamera2 pick the
  1536x864 sensor mode, which reads only the middle of the sensor. After the 4:3 cut,
  that used 50% of the sensor width and 67% of its height. The 2304x1296 mode reads
  the whole sensor at 56 fps; output 640x360 keeps the sensor's 16:9 shape.
- Side, USB webcam. 640x480, 1280x720 and 1280x1024 are crops from the middle;
  1024x768 shows the full view (same as 2048x1536) at 19 fps. side_bgr() shrinks it
  to 640x480. The outer corners are dark because the lens's image circle is smaller
  than the sensor; ignore them in detection.

Lane centering (track_vision, quiz3_autonomous, lane_app, calibrate_track) still uses
its own 640x480 front setup. Its calibration was measured on that cropped view, so
switch it to front_config() only together with a recalibration.
"""

import cv2

FRONT_SENSOR_MODE = (2304, 1296)
FRONT_SIZE = (640, 360)
SIDE_CAPTURE_SIZE = (1024, 768)
SIDE_SIZE = (640, 480)


LEGACY_FRONT_SIZE = (640, 480)


def front_config(camera, legacy=False):
    """Picamera2 configuration for the full-view front camera.

    legacy=True returns the old cropped setup (640x480, sensor mode left to Picamera2),
    which the existing lane-centering calibration was measured on.
    """
    if legacy:
        return camera.create_video_configuration(
            main={"size": LEGACY_FRONT_SIZE, "format": "XRGB8888"}, queue=False)
    main = {"size": FRONT_SIZE, "format": "XRGB8888"}
    try:
        return camera.create_video_configuration(
            main=main, sensor={"output_size": FRONT_SENSOR_MODE, "bit_depth": 10}, queue=False)
    except TypeError:  # older Picamera2 without the sensor= argument
        return camera.create_video_configuration(
            main=main, raw={"size": FRONT_SENSOR_MODE}, queue=False)


def side_config(camera):
    """Picamera2 configuration for the full-view USB side camera."""
    return camera.create_video_configuration(
        main={"size": SIDE_CAPTURE_SIZE, "format": "XRGB8888"}, queue=False)


def front_bgr(array):
    """Front camera array -> 640x360 BGR."""
    return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)


def side_bgr(array):
    """Side camera array -> 640x480 BGR (the USB camera's channels arrive as RGBA)."""
    frame = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if (frame.shape[1], frame.shape[0]) != SIDE_SIZE:
        frame = cv2.resize(frame, SIDE_SIZE, interpolation=cv2.INTER_AREA)
    return frame
