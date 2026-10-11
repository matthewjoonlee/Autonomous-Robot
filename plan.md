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
- The Quiz 3 track is continuous, with no floor gaps. The first live diagnostic used only part of the track; that setup does not represent the intended course.

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

### Initial IMU results (October 5, 2026)

- Pulled two physical-test CSVs into `imu_logs/`. Neither contained unavailable readings (584 readings total).
- The second log, `imu_20261005_170244_281459.csv`, contains the complete left/center/right/center sequence. Final three-second readings were: initial centered 0.00 degrees, left -24.75 degrees, returned centered -1.25 degrees, right +28.938 degrees, final centered -0.688 degrees.
- Left is negative and right is positive with the existing `SIGN = +1`; retain that convention.
- Initial stationary readings after zeroing were stable. Return offsets below approximately 1.3 degrees are encouraging but cannot distinguish sensor drift from manual alignment error. A longer stationary drift test remains useful before tuning heading control.
- The first log ends during the left phase and does not validate right rotation or return-to-center behavior.
- Blinka initially failed to open GPIO devices with two-digit chip numbers. The test ran after the library-upgrade step; use a Blinka release containing the multi-digit chip-selection fix (9.2.0 or later) on this Pi.
- Proceed to labeled camera calibration. Physical steering response and autonomous recovery remain untested.

## Phase 2: camera calibration and perception without motors

Implement a camera calibration/perception tool before powered autonomous driving.

`calibrate_track.py` now implements stationary collection through the remote shell, without display-window or motor dependencies. Run `python calibrate_track.py`, align centered and straight, and enter `z` to set the heading reference. Position the robot before entering each capture command: `c` centered straight, `l` left offset straight, `r` right offset straight, `a` centered rotated left, `d` centered rotated right, or `x` combined. Enter an approximate displacement/angle note when prompted. Capture at least two images per pose, keep the same IMU reference, and use `q` to finish. Images and a CSV with bracketing IMU readings and capture metadata are saved together under `track_calibration/session_*/`. These are nearby readings, not hardware-synchronized camera/IMU samples. Perception estimation is still to be implemented and validated against these labeled images.

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

### Initial vision implementation and results

`track_vision.py` implements offline and live diagnostics, with no motor commands. It uses a lower-image yellow mask, dash geometry filtering, and robust straight-line fitting. Its confidence score is a heuristic quality measure, not a probability. It reports normalized image-position and image-direction errors separately from relative IMU heading; these are not physical lateral distances or motor commands.

Run `python track_vision.py` to analyze the five labeled captures from `session_20261005_172428_111633`. Run `python track_vision.py --live --duration 60` on the Pi to collect a one-minute motor-free diagnostic session. Stop other camera programs, keep motors disabled, align centered and straight on a continuous section with several nearby dashes visible, and press Enter. By default, live mode captures a fresh centered reference and refuses to begin if that line is not reliably detected. Use `--reference` only to deliberately retain a supplied centered reference. Move the robot by hand through the same poses. The live tool saves unannotated PNGs in `raw/`, annotated JPGs, the startup reference, and CSV readings approximately once per second under `vision_results/`; `Ctrl+C` also stops it. Cyan is the calibrated reference and green is the detected line. The IMU is read near each frame rather than hardware synchronized.

Offline results on the five labeled captures:

| Pose | Status | Image position error at 85% image height | IMU relative heading |
| --- | --- | --- | --- |
| Centered straight | Tracked | 0.0 px (reference itself) | 0.00 degrees |
| Left offset straight | Tracked | +62.2 px (line right of reference) | -0.875 degrees |
| Right offset straight | Tracked | -52.1 px (line left of reference) | -0.0625 degrees |
| Centered rotated left | Line lost | No estimate | -13.875 degrees |
| Centered rotated right | Tracked | -272.3 px | +13.125 degrees |

The large image shift during rotation confirms that pixel error alone must not be interpreted as sideways displacement. No distance calibration is inferred from the offset labels because the measurement notes were left blank. Live validation, wider displacement/rotation coverage, and repeated centered references are still needed. The left-rotation loss is correctly flagged; autonomous behavior on loss remains a stop until a separately bounded reacquisition procedure is implemented and tested.

All three original centered front snapshots also produce tracked lines, but their image errors relative to the new centered reference are +20.7 px, +26.0 px, and -2.3 px. This spread means the single-reference 16 px diagnostic deadband is provisional and should not become an autonomous tolerance. Collect repeated carefully aligned centered poses with the mounting unchanged before setting a control deadband or physical centering tolerance.

Regression checks cover the labeled physical images plus blank images, background yellow, horizontal crossing markings, and competing forward paths. Run `python -m unittest discover -s tests -p 'test_track_vision.py'` in an environment with OpenCV and NumPy. These checks do not establish autonomous recovery performance or general intersection handling.

### First live diagnostic run: revision required

`vision_results/run_20261005_173524_741537` contains 60 annotated frames and 60 CSV rows. Status counts are 12 tracked, 28 line lost, and 20 ambiguous. All 60 IMU readings were available. The first approximately 22 seconds have zero relative heading but inconsistent vision statuses, so the current vision implementation is not ready for motor control.

Visual inspection of frames 0, 5, 17, 40, 45, and 59 shows a floor gap between the nearby track and a separate farther track piece. Several frames contain only one nearby dash. Some accepted fits incorrectly join that dash with small yellow-mask speckles on green surfaces. A tracked status therefore does not establish a correct fit in this run.

Before powered tests, strengthen candidate filtering and require plausible track support so grass speckles cannot form a path. Save raw frames alongside annotations in future live runs; the current annotations change image pixels and must not be treated as clean raw inputs for detector revalidation. Confirm whether the floor gap is part of the Quiz 3 setup; if so, specify explicit gap behavior instead of silently fitting across separate track pieces. Repeat live diagnostics on the intended setup and verify every accepted fit visually before steering calibration.

### Detector revision and next physical test

The user confirmed that the actual track has no gaps. The revised detector narrows the yellow hue range to exclude the green tail, rejects sparse blobs, requires dark track around each candidate dash, and checks dark track support along both sides of the fitted path. These are initial thresholds based on the supplied images, not guarantees across lighting conditions. It retains the existing minimum dash count and near-field coverage requirements rather than extrapolating a path from one dash.

Ten regression checks passed after this change, including the original and labeled physical snapshots, collinear yellow speckles on green plus one nearby dash, bright-floor markings, floor-gap rejection, competing paths, and simulated live capture with exact unannotated PNG saving and hardware cleanup. This does not establish real-time performance or physical continuous-track reliability.

Repeat the 60-second stationary diagnostic on a continuous straight section, keeping several nearby dashes in view. Hold centered for approximately 10 seconds, left offset for 10 seconds, right offset for 10 seconds, left rotation for 10 seconds, right rotation for 10 seconds, and centered again for the remainder. Start with small offsets and rotations, then increase after reviewing the results. Capture notes on approximate distances/angles if possible. Upload the entire new run folder including `raw/`, both startup reference images, and `results.csv`. Review accepted fits on the actual yellow line and inspect line-loss cases before enabling motors.

### Continuous-track live results (October 5, 2026)

`vision_results/run_20261005_174624_462341` contains 60 raw PNGs, 60 annotated frames, startup references, and a CSV. It reports 54 tracked frames, six line-loss frames, zero ambiguous frames, and no unavailable IMU readings. Replaying all raw frames locally reproduced every logged detection status.

Representative annotated frames show the accepted fits following the yellow line. Initial centered frames stay within approximately 0.3 px of the startup reference. Offset holds show approximately +96 px and -55 px errors; heading changed slightly too, so these are not pure physical-distance measurements. A right rotation around +14 degrees remains tracked. Frames 18–21 lose the near-field line around -15 to -16 degrees left rotation; inspected frame 20 places the line outside the right edge. Frames 22 and 27 lose tracking during visibly blurred pose transitions. The last 31 settled frames are all tracked, with image error between +5.3 and +5.7 px and heading +0.375 degrees.

This is substantially more stable than the partial-track test and supports proceeding to bounded low-speed steering calibration, not unrestricted autonomous recovery. The left-camera visibility limit and motion-blur loss still require explicit handling. Maintain stop-on-loss in the initial autonomous controller; any heading-based reacquisition mode must be bounded and physically tested separately. The one-second diagnostic sampling rate does not establish performance at the eventual motor-control rate.

## Phase 3: steering calibration at low speed

Implement a bounded motor-test mode with explicit operator start and a fixed short duration for each command.

`calibrate_motors.py` now implements one trial per invocation: `stop`, `forward`, `left`, or `right`. Without `--execute` it only prints a command preview and does not import hardware libraries. Physical execution initializes the IMU and motor controller, checks that the serial port is open, requests zero motor commands, configures the 200 ms timeout and ramping 8 (matching manual control), and requires typing `RUN` before the trial. Ramping is a persistent hardware setting.

Initial settings are speed command 20, steering magnitude 6, and a 0.5-second command pulse. These are uncalibrated command values, not a measured speed or guaranteed low physical speed. Each run stops at the deadline and observes heading for another 0.5 seconds with repeated stop commands. Limits are speed <=35, steering <=15, duration <=2 seconds, and a default 15-degree heading-change abort. Sensor/loop delays over 100 ms or unavailable heading abort rather than resume driving. `Ctrl+C`, termination, and hangup signals request stopping; a remote-shell disconnect is not guaranteed to deliver a signal. The motor timeout must still be verified on hardware.

First physical test sequence, one command at a time:

```bash
python calibrate_motors.py --trial stop --execute
python calibrate_motors.py --trial forward --execute
python calibrate_motors.py --trial left --execute
python calibrate_motors.py --trial right --execute
```

Stop other motor-control programs, clear the area, and keep a physical stop accessible. Verify zero commands with the stop trial; then run forward and confirm the robot stops before proceeding to steering. Reposition straight between trials. If motion is unexpected or stopping fails, end testing rather than execute the next command. Record approximate distance, actual turn direction, and stopping behavior at the note prompt. Speed 20 may not overcome motor deadband; report this rather than immediately increasing several parameters. The tool saves each trial's CSV and operator note under `motor_calibration/trial_*/`. Upload completed trials for analysis.

Seven simulated regression checks passed for direction/deadline behavior, missing sensor readings after motion, heading limits, interruption after motion, delayed loops, slow sensor reads, and logging failure. Run `python -m unittest discover -s tests -p 'test_calibrate_motors.py'`. These checks establish software behavior only; physical stopping, serial timeout, direction, and steering strength remain unvalidated.

### Initial physical motor results (October 11, 2026)

Five trial logs were pulled. The first three are `stop` trials with speed/turn commands `(0, 0)`; their lack of motion is expected, not evidence that forward command 20 failed. No forward-command-20 trial is present in this upload.

The last two trials command forward speed 25, turn 0 for approximately one second. The final trial's operator note is `forward 11 cm`; its peak relative heading was +0.25 degrees and final heading +0.0625 degrees. The earlier forward trial has no movement note and ends at +0.5 degrees. Neither forward trial logged an abort. These logs contain requested commands and heading, not wheel-speed measurements or acknowledgment that the motor hardware applied each command.

Forward command 25 is a usable initial calibration setting based on the observed approximately 11 cm movement. Do not infer precise speed, stopping distance, or repeatability from one noted distance. Confirm prompt physical stopping, then collect left and right trials separately using `--speed 25 --turn 6 --duration 0.5`, repositioning straight between trials. Steering response and the hardware timeout remain unverified.

### Initial steering direction results (October 11, 2026)

- `trial_20261011_181714_081916`: labeled left, commanded speed 25 / turn -6 for 0.5 seconds. Heading increased to +1.3125 degrees at stop and settled to +1.25 degrees. Operator noted that it appeared to turn right.
- `trial_20261011_181804_124400`: labeled right, commanded speed 25 / turn +6 for 0.5 seconds. Heading decreased to -0.1875 degrees at stop and settled to -0.3125 degrees. Operator noted a slight left turn.
- These signs and observations suggest the physical steering convention is opposite the comments in `sabertooth.py`. The positive-turn response is small, so repeat verification before changing the interface or relying on a correction sign. Keep the validated IMU convention (positive right / negative left) unchanged.
- Next bounded verification: retain speed 25 and duration 0.5 seconds, increase only steering magnitude to 10, and run the labeled left and right trials separately with straight repositioning. Record actual direction and stopping. If the reversal is repeatable, translate desired physical steering into Sabertooth commands explicitly in the future autonomous controller; do not silently alter manual-control behavior.

### Steering reversal verified (October 11, 2026)

Three additional trials used speed 25 and steering magnitude 10 for 0.5 seconds. Raw turn -10 settled at +1.5625 degrees and +3.125 degrees in two trials; the second operator note says `definitely moved slightly right`. Raw turn +10 settled at -1.125 degrees; its note says `moved to the left`. All trials completed without a logged abort.

Use `sabertooth_turn = -desired_physical_turn` in the separate autonomous controller, with desired physical right positive and left negative. Preserve the validated IMU sign and existing manual-control behavior. The responses are unequal and variable, so do not assume symmetric steering gains or derive a precise model from these short pulses. Begin the next controller with bounded duration, forward command 25, conservative steering limits, and stop-on-line-loss; validate centered driving before offset recovery and foot pushes. Operator notes confirm turn direction but do not explicitly quantify physical stopping or verify the serial timeout.

1. Verify stop behavior before forward motion.
2. Establish a low forward speed that moves reliably on the track.
3. Apply brief left and right steering commands while moving forward.
4. Record command values, IMU response, visible motion, and response delay.
5. Establish minimum effective steering, forward drift, conservative steering limits, and suitable ramping.
6. Verify `Ctrl+C`, exception cleanup, and the configured motor serial timeout.

Pass condition: command directions match sensor directions, corrections are repeatable, and stopping works. Existing working manual control does not establish autonomous steering gains.

## Phase 4: autonomous controller

Implement `quiz3_autonomous.py` with separate perception and control logic plus a configuration file or command-line settings for calibrated values.

### Initial bounded controller implemented

`quiz3_autonomous.py` now implements a short straight-track test. Its default mode reads camera/IMU data and logs proposed commands without importing or initializing the motor controller. Physical motor commands require `--execute` and typing `RUN`. Default duration is three seconds, with a maximum of ten seconds for this initial version. Capture a fresh reference while the robot is physically centered and straight; it must pass tracking/confidence checks before the run begins. Keep that reference and IMU zero fixed during the run.

The initial feedback law uses near-field normalized image error to request a heading capped at +/-4 degrees. Relative IMU heading regulates the turn toward that target. Physical positive-right steering is explicitly inverted into negative Sabertooth turn commands. Forward command is 25 and raw steering magnitude is capped at 10. Position gain 40 degrees per normalized image error, heading gain 3 turn units per degree, and an 8 px image deadband are provisional starting values, not physically tuned gains. Image error includes perspective changes from rotation; it is not interpreted as a measured lateral distance.

The controller stops and exits on line loss/ambiguity, confidence below 0.65, unavailable IMU data, camera/processing or command-loop delays over 150 ms, heading reaching +/-10 degrees, runtime errors, or interruption. The motor controller requests a 200 ms serial timeout and ramping 8. There is no automatic rearm or blind line reacquisition. Sample freshness is checked through capture/processing timing with camera queueing disabled, not a hardware-synchronized camera/IMU timestamp.

Avoid disk encoding and console output in the command loop: retain telemetry and selected raw frames in memory, request a motor stop, then save `readings.csv`, `outcome.json`, raw frames, and startup reference files under `centering_results/run_*/`. A forced kill can lose buffered telemetry; a remote-shell disconnect is not a guaranteed stop. Verify actual stopping and the hardware timeout separately.

First test sequence on the Pi, with the robot centered and straight on continuous track:

```bash
python quiz3_autonomous.py --duration 3
```

This is motor-free. If it completes without a fault and reports proposed near-zero corrections, proceed to one powered run with no pushes and a physical stop ready:

```bash
python quiz3_autonomous.py --execute --duration 3
```

Both runs require capturing the centered reference and typing `RUN`. Do not execute the powered trial if the diagnostic run reports a timing/tracking fault. Stop other motor-control programs and cameras. Record travel, drift, correction direction, and prompt stopping at the operator-note prompt. Upload results using:

```bash
git add -- centering_results/
git commit -m "Add initial bounded centering results"
git push origin main
```

Nine new simulated tests passed for physical steering direction/sign conversion, straightening feedback, command limits, invalid/low-confidence/stale data, heading limits, line-loss latch, IMU loss/interruption, capture delays, run deadlines, and motor-free operation. Physical centering performance remains unvalidated. Review these first results before intentional offsets or foot pushes.

### Motor-free controller validation (October 11, 2026)

`centering_results/run_20261011_183051_983741` completed the requested three seconds with `execute: false`. All 74 readings were tracked and diagnostic-only; proposed steering was zero throughout. Image error stayed between approximately 0.0 and 0.1 px, relative heading stayed zero, and processing took 12.2–48.6 ms (mean approximately 29.0 ms). Maximum time between logged samples was 69.4 ms, below the configured 150 ms limit in this stationary test.

The operator note says `Didn't really move`; no motor controller was initialized in this mode, so no movement was expected. Startup and selected raw frames show a continuous straight track. Proceed to one three-second powered run from centered and straight, without offsets or pushes, then review travel/stopping and logged correction behavior. Timing while moving and actual centering performance are still unvalidated.

### First powered controller result: coverage threshold failure

`centering_results/run_20261011_183304_621915` enabled motors but stopped after approximately 0.727 seconds with `Tracking fault: line_lost`, before the requested three-second duration. Fifteen samples commanded forward 25; raw steering was zero except two samples at -1. Heading stayed within -0.1875 to +0.125 degrees, image errors within approximately -3.0 to +3.5 px, and processing remained below 46 ms. No operator note was saved because the fault exited before that prompt; prompt physical stopping still needs operator confirmation.

The final raw frame visibly retains the yellow line. Reinspection finds seven accepted collinear dash centers with full dark-track support, but their vertical span is approximately 0.167 of image height, below the detector's fixed 0.18 requirement. Some other dashes are excluded by the width/height rule near a tile seam. This is a detector coverage rejection while moving, not evidence that the robot departed the line or that correction gains are wrong. Revise and regression-test coverage/geometry handling against this real failure frame before repeating a powered run; retain stop-on-loss behavior.

### Powered-frame coverage fix

The operator confirmed that the robot physically stopped promptly after the first powered fault. This verifies stopping in that observed case, not the independent hardware timeout.

The detector now accepts a shorter span (at least 0.14 image height) only when at least six accepted dashes support the fit and their vertical extent contains the near-field measurement row. Sparse fits retain the 0.18 minimum. Dark track support, geometry filtering, ambiguity rejection, and stop-on-loss remain enforced. Confidence weights the stronger near-field support so a valid compact fit is not immediately rejected by the controller's unchanged 0.65 threshold.

The former failure frame now tracks at approximately 0.904 confidence with near-zero reference error. All four stored raw frames from the powered trial pass the controller's vision checks. Replaying the previous 60-frame continuous-track run preserves all tracked/line-loss classifications. All 29 regression checks passed, including the real failure frame and rejection of sparse compact paths and tightly clustered speckles.

Repeat one three-second powered run from centered and straight, with no pushes, using `python quiz3_autonomous.py --execute --duration 3`. Upload `centering_results/` after recording movement and stopping observations. The software fix is validated on saved frames; moving performance beyond the previous 0.73 seconds still needs physical verification.

### Centered powered retest completed

`centering_results/run_20261011_183715_697962` completed the requested three seconds with motors enabled. All 63 sampled frames were tracked (62 driving rows and one duration-complete row). Image error ranged from -9.7 to +2.9 px and ended at -3.2 px; relative heading stayed between -0.1875 and +0.25 degrees. Raw steering ranged from -1 to +2, processing peaked at 50.1 ms, and confidence stayed above 0.835. The operator noted `moved forward straight slowly`.

This validates a short centered forward trial, not recovery from displacement. Next run one bounded five-second trial starting approximately 2 cm left of the line: capture the reference and IMU zero while centered/straight, then shift the robot left without rotating it while paused at the `RUN` prompt. Do not recapture or zero from the displaced pose. Record the starting offset, correction direction, final position, and stopping. Review that trial before testing the opposite offset or introducing pushes.

### Four five-foot trials requested

The user requested moving directly to a full four-trial test: centered forward, left-offset recovery, right-offset recovery, and recovery from one foot nudge. `quiz3_course.py --execute` runs these trials in that order with a stop, saved logs, and explicit setup between each. A fault, cancellation, or interruption ends the suite. The same feedback controller is used; successful recovery is not assumed from the centered test.

Mark a start and finish five feet (152.4 cm) apart on continuous straight track. Existing hardware interfaces provide no wheel-distance feedback. At speed command 25, the earlier 11 cm forward observation gives a provisional nominal travel time of approximately 13.9 seconds. Course mode provides a maximum approximately 27.7-second fallback time budget, capped at 30 seconds, and requires the operator to press Enter at the actual finish. The timer is not an odometer: a time-limit result does not confirm five-foot completion. `outcome.json` records the trial label, requested distance, timing estimate, stop reason, and `distance_measured: false`.

For every trial, return to the start centered/straight and capture the reference before introducing an offset. Straight: leave centered, type `RUN`, and press Enter at the finish. Left/right: after the reference is captured, shift approximately 3 cm in the named direction without rotation, then type `RUN`. Nudge: start centered and apply one small sideways foot nudge after motion begins. Do not re-zero or capture the reference from the offset position. Keep a physical stop ready and stop other camera/motor programs. Large heading changes or line loss still stop this version; unvalidated blind reacquisition was not enabled.

Course mode saves raw frames approximately once per second plus the final frame while retaining telemetry for every sample. Record starting displacement/nudge, visible correction, final position, travel distance, and stopping in each operator note. Startup reference files and run labels allow comparing all four runs without another code edit.

Pi commands:

```bash
cd ~/code
git pull --ff-only origin main
python quiz3_course.py --execute
```

After the trials finish (or a fault stops the suite), upload:

```bash
git add -- centering_results/
git commit -m "Add four five-foot lane centering trials"
git push origin main
```

The four-trial order, stop after cancellation/fault, time-estimate conversion, and finish-stop checks before/during capture are covered by five new simulated tests. These supplement the existing controller, vision, and motor checks; physical offset and nudge recovery remain to be measured.

### First course-suite attempt

Only the straight trial was uploaded: `centering_results/run_20261011_184441_221716`. It stopped at approximately 0.471 seconds on `Tracking fault: line_lost`, so the suite did not reach left/right/nudge trials. The first ten samples commanded forward 25 / turn 0; heading stayed within +0.125 degrees and image error within approximately -3.2 to +0.7 px. The last raw frame still visibly contains the line with modest motion blur.

The failure frame retains six accepted dashes, but their span is approximately 0.130 image height, below the new dense-fit 0.14 requirement. Other true dashes were rejected at width/height ratios around 3 or greater, including dashes split near a tile seam. The count/span and individual-shape rules still depend too strongly on dash phase and blur during motion. A broader motion/phase regression dataset and robust handling of short tracking interruptions are needed before the full suite can succeed. Any interruption recovery should stop forward commands while waiting for trustworthy new frames and remain bounded; do not continue blind forward motion or merely reduce controller confidence globally.

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
3. Keep motors disabled during stationary sensor tests. `test_imu.py` is implemented and the initial physical left/right heading test is recorded above.
4. Run `python test_imu.py` in the Pi's existing Python environment through Raspberry Pi Connect. Wait for readings to settle, align straight, and enter `z` followed by Enter. Hold still for 10 seconds, enter `l` and rotate left approximately 20–30 degrees, enter `c` and return straight, then repeat with `r` to the right and `c` to return. Hold each pose for a few seconds. Enter `q` or press `Ctrl+C` to finish. Share the readings and generated CSV from `imu_logs/`.
5. Then collect the labeled offset and rotation snapshots from Phase 2 before attempting autonomous motion.

No autonomous code, gains, or recovery performance are validated by this plan alone.
