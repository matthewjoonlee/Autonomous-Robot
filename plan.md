# Quiz 3 implementation plan

## Objective and confirmed setup

Drive forward on a straight track while keeping the robot's center over the yellow dashed line. Recover when the instructor pushes the robot sideways, rotates it, or does both.

- The robot straddles the yellow line; the line is the desired center of the robot.
- The front camera is mounted on the front left of the robot.
- All three `snapshot_*_front.jpg` images show the robot centered and pointing straight along the track. The yellow line is therefore expected to appear right of the image center.
- The side snapshots are too dark to use for this initial implementation.
- Motors work; the IMU has not yet been tested.
- Robot body width: 20.5 cm. Overall width including wheels: 28.6 cm. Track width: approximately 32 cm.
- Wheels may travel onto the adjacent green surface during recovery. The goal remains to return the robot's center to the yellow dashed line. The approximately 1.7 cm margin per side on the black track, `(32 - 28.6) / 2`, is not a hard recovery limit or a reason to stop by itself.
- Launch a separate autonomous program through Raspberry Pi Connect's remote shell. Keep `robot_controller.py` as the PS5 manual-control program.
- Quiz 3 covers straight sections. Curves and autonomous intersection decisions are outside this first version. Crossing markings must not silently become a new tracking target.

## Existing interfaces

- `sabertooth.py`: `Sabertooth.drive(speed, turn)`, `stop()`, and `close()`. Verify steering direction on the physical robot despite the documented positive-right convention. The constructor currently requests a 200 ms serial timeout.
- `imu_controller.py`: `IMUDevice.zero()`, `heading()`, `delta()`, and `close()`. The BNO055 is configured for gyro/accelerometer fusion without a magnetometer. Treat its heading as a relative reference and measure drift during testing. Raw gyro angular velocity is not currently exposed by this wrapper.
- `test_display.py`: working front-camera discovery, capture, color conversion, and snapshot handling. Reuse its camera conventions without running its demonstration display loop or initializing an unnecessary side camera.

## Phase 1: stationary IMU test

Implement `test_imu.py` first. It must never initialize the motor controller or send motor commands.

1. Print timestamped heading and relative heading readings, including unavailable readings.
2. Allow the operator to zero the heading while the robot points straight along the track.
3. Hold still, rotate left, return straight, rotate right, and return straight.
4. Record the sign of each rotation, approximate angle, update reliability, and drift while stationary.
5. Exit cleanly with `Ctrl+C` and close the IMU.

Pass condition: readings are usable, rotation direction is understood, and drift/dropouts have been measured. Do not select controller gains before seeing these results.

## Phase 2: camera calibration and perception without motors

Implement a camera calibration/perception tool before powered autonomous driving.

Collect labeled front-camera snapshots with the camera mounting unchanged:

| Position | Orientation | Purpose |
| --- | --- | --- |
| Centered | Straight | Confirm the existing reference |
| Left of line | Straight | Measure left displacement response |
| Right of line | Straight | Measure right displacement response |
| Centered | Rotated left | Separate rotation from displacement |
| Centered | Rotated right | Separate rotation from displacement |
| Left/right of line | Rotated | Check combined disturbances |

Record approximate displacement and rotation for each pose, preferably with synchronized IMU readings. Include poses with wheels on the green surface, since these are permitted recovery states. Measure the usable surrounding surface and robot length to understand clearance for displaced and angled poses.

Perception approach:

1. Restrict processing to the track region of the front image, excluding the classroom background.
2. Segment yellow markings and reject implausible blobs using geometry and location.
3. Fit the forward dashed-line path across multiple image rows, bridging ordinary dash gaps. Reject horizontal crossing markings as forward-path candidates.
4. Compare near-field line position and line direction with the centered reference. Account for perspective and camera offset; do not simply target half the image width or assume one pixel offset works at every row.
5. Produce signed lateral error, visual alignment error, a confidence score, and a capture timestamp. Until physical calibration supports distance units, report normalized/image errors rather than centimeters.
6. Show an annotated preview or save annotated frames so the operator can verify the selected line and errors.

Pass condition: all labeled poses produce the expected error directions, centered frames produce small errors, and missing/ambiguous markings produce low confidence. Review all three supplied front snapshots offline. Collect additional examples if they disagree.

## Phase 3: steering calibration at low speed

Implement a bounded motor-test mode with explicit operator start and a fixed short duration for each command.

1. Verify stop behavior before forward motion.
2. Establish a low forward speed that moves reliably on the track.
3. Apply brief left and right steering commands while moving forward.
4. Record command values, IMU response, visible motion, and response delay.
5. Establish minimum effective steering, forward drift, conservative steering limits, and suitable ramping.
6. Verify `Ctrl+C`, exception cleanup, and the configured motor serial timeout.

Pass condition: command directions match sensor directions, corrections are repeatable, and stopping works. Existing working manual control does not establish autonomous steering gains.

## Phase 4: autonomous controller

Implement `quiz3_autonomous.py` with separate perception and control logic plus a configuration file or command-line settings for calibrated values.

Control behavior:

1. Start stopped. Require fresh, confident camera estimates and usable IMU data before enabling motion. Zero the IMU while aligned straight.
2. Use camera lateral error to request a bounded corrective heading relative to the straight track reference.
3. Use IMU relative heading to regulate the turn toward that requested heading. Incorporate visual alignment to check the heading reference and limit the effect of drift.
4. Reduce the corrective heading as lateral error decreases so the robot straightens over the line. Heading hold alone cannot detect a sideways push.
5. Begin with proportional feedback and, if measurements justify it, damping. Avoid integral action initially. Set gains only after calibration.
6. Clamp steering, limit changes between commands, and reduce forward speed during larger corrections. Keep the initial version moving forward rather than adding reverse recovery.
7. Use monotonic time and measure actual sensor/control rates. Send commands often enough for the configured motor timeout, without replaying stale frames or allowing a command backlog.

Stop and lifecycle behavior:

- Stop when camera tracking is lost/ambiguous beyond a short validated tolerance, sensor data becomes stale, or the IMU becomes unavailable.
- Stop on `Ctrl+C`, termination signals, initialization failure, and runtime exceptions. Stop motors before closing hardware resources.
- Validate hardware initialization rather than assuming constructors succeeded.
- After a tracking fault, remain stopped until the operator explicitly restarts/rearms.
- Preserve and physically verify the motor-controller timeout as a backup when command delivery stops.
- Do not assume disconnecting Raspberry Pi Connect stops the process. Verify remote-shell behavior and arrange an accessible physical stop/power control for powered tests.
- Run only one motor-control program at a time; do not run `robot_controller.py` concurrently.

Log timestamps, frame age, perception confidence, lateral/alignment errors, IMU heading, desired heading, speed/turn commands, operating state, and stop reason. Save selected frames around disturbances for diagnosis.

## Phase 5: recovery testing and tuning

Test in this order at low speed, with space to recover and an operator ready to stop. Begin with small offsets, then include wheels crossing onto the permitted green surface. Set recovery limits from available surface, obstacles, and reliable line visibility rather than the black-track width alone.

1. Drive from centered and straight with no disturbance.
2. Start slightly left, then slightly right, while pointing straight.
3. Start centered but rotated left, then right.
4. Combine an offset and rotation.
5. Apply small sideways pushes while driving, then pushes that also rotate the robot.
6. Increase disturbance size only after repeatable recovery.
7. Test missing line, ambiguous crossing markings, unavailable camera/IMU, and interrupted command delivery.

Tune one parameter at a time. Measure peak displacement, recovery time/distance, final alignment, oscillation, and stop response. Agree on numeric acceptance limits after measuring the usable recovery area and the instructor's expected push size.

Success means the robot returns to the yellow centerline, aligns with the straight track, continues forward without sustained oscillation, and stops predictably when it cannot track safely. A push beyond the visible/usable recovery area should cause a stop rather than blind steering.

## What to do next

1. Place the robot centered and straight on the track. Keep the camera mount unchanged.
2. Widths and permission to recover on the green surface are recorded. Measure robot length and the usable surrounding surface, and note whether the camera points straight ahead or is angled.
3. Keep motors disabled for the first sensor tests. `test_imu.py` is implemented; physical IMU validation is still pending.
4. Run `python test_imu.py` in the Pi's existing Python environment through Raspberry Pi Connect. Wait for readings to settle, align straight, and enter `z` followed by Enter. Hold still for 10 seconds, enter `l` and rotate left approximately 20–30 degrees, enter `c` and return straight, then repeat with `r` to the right and `c` to return. Hold each pose for a few seconds. Enter `q` or press `Ctrl+C` to finish. Share the readings and generated CSV from `imu_logs/`.
5. Then collect the labeled offset and rotation snapshots from Phase 2 before attempting autonomous motion.

No autonomous code, gains, or recovery performance are validated by this plan alone.
