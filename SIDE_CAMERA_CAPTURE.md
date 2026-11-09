# Compare side-camera positions

On the Pi, stop any other camera/control programs, keep motor power off, and run:

```sh
python capture_side_camera.py
```

The tool automatically selects a uniquely identified USB/UVC camera. If it cannot
identify one, use the camera list it prints and rerun with its Picamera2 index:

```sh
python capture_side_camera.py --camera 1
```

1. Give the mounting position a label, such as `straight_down`.
2. Press Enter to capture. Label the robot's position relative to the lane, add any
   measurements, and press Enter again once the robot is still.
3. Move the robot by hand and repeat for centered/straight, left offset, right
   offset, before a turn, quarter turn, halfway through, and aligned after the turn.
4. Enter `m` to change the mounting position. Try a slight outward tilt, recording
   its approximate angle and camera height. Repeat the same robot poses at the same
   track locations, including the same turning direction.
5. Enter `q` to finish. Share the ZIP path printed by the program.

For comparison, keep lighting and camera height consistent unless height is what
you are testing. Record the actual pose yourself: the program labels photos but
does not determine whether a turn is complete. Include a capture where the robot
has reached the turn's final heading but is still sideways from its desired lane
position, so we can assess both alignment and position.

Each session contains unannotated 640×480 PNG images and `captures.csv` with the
full labels, notes, camera index, and available exposure metadata. Automatic
exposure remains enabled as configured by the camera; no display enhancements
are applied. The program never opens or commands the motor controller.
