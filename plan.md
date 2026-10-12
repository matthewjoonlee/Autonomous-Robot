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

### Motion-blur handling implemented

Three changes address the course failure together:

1. The yellow mask closes tiny vertical splits before extracting candidates. Dash geometry allows moderate flattening (width/height <=5 and fill >=0.45) while retaining yellow hue, dark surrounding track, dark support along the fitted path, and competing-path rejection. Six or more near-field dashes can support a span >=0.12 image height; sparse fits still need >=0.18. Both saved powered failure frames now track at approximately 0.97 confidence. The previously recorded moving frame 6 from the continuous-track diagnostic is now rejected; bounded reacquisition handles an unusable frame rather than forcing every blurred image to produce a fit.
2. On line loss, ambiguity, or confidence below threshold, the controller immediately requests a motor stop and enters `reacquiring`. It resumes only after three consecutive trustworthy frames. Persistent loss for one second or more, more than five interruption episodes, unavailable IMU, stale data, excessive heading, or other errors end the run. Tracking waits retain the original reference/IMU zero and do not command forward movement or blind heading turns. The course time budget includes these pauses.
3. Camera initialization requests 5000 microseconds (5 ms) exposure with manual gain estimated from settled auto-exposure metadata, bounded by hardware limits and a gain cap of 8. Explicit manual exposure/gain modes are set where available. Requested/observed exposure and gain are printed and saved in `outcome.json`. Short exposure may require more illumination; increased gain adds noise and cannot compensate arbitrarily dim lighting. Actual shutter/brightness on this Pi remains to be validated. Both programs accept `--exposure-us` (0 retains auto exposure).

Picamera2 exposure controls and metadata follow the official manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf. Thirty-nine regression checks pass, covering the saved blurred failure, persistent vs transient tracking loss, no motion during reacquisition, repeated-interruption bounds, gain limits, and auto-exposure override as well as the existing controller/vision/course checks. Keep speed command 25 for the next suite; higher-speed operation has not been validated.

Rerun the four-trial program on the Pi with `python quiz3_course.py --execute`. Mark the five-foot finish and press Enter there, capture each reference centered/straight, then follow the left/right/nudge setup prompts. Upload `centering_results/` using the previously documented commit/push commands. Review observed camera exposure and recovery events alongside motion before increasing speed.

### Lane-centerline tracking and tolerant control (superseded stop-on-blur)

Stopping on every unusable frame made motion jerky and, bounded at five interruptions, would likely end a 14-second run. Dash-shape thresholds kept failing under new blur and dash-phase combinations. The controller now tracks the dark lane instead of individual dashes and keeps driving through short gaps.

1. `track_vision.detect_lane` scans rows in the same region of interest. Dark pixels plus yellow form the lane mask, so dashes, including rotation-smeared dashes, do not split it. Each row contributes the midpoint of the dark run nearest the predicted center: the previous fit while it is recent, otherwise bottom-up continuity from the widest run. Runs cut by the image border use the dash pixels inside them. Runs over 70% of the width (crossing bands) are ignored. A robust line fit gives the centerline. Confidence is row coverage times fit quality, and failed rows are skipped rather than rejecting the frame. On all 83 saved frames where both detectors track, the lane near-row position agrees with the dash fit within 1.9 px; it also tracks every powered-run frame without a prior and three heavy-blur/rotated diagnostic frames the dash detector rejects. It runs about 1.0 ms per frame on a Mac versus 1.5 ms for the dash detector.
2. `LaneTracker` low-pass filters the near-row image error (weight 0.5 per new frame). Unusable frames leave the estimate in place, relaxing toward straight with a 0.5 s time constant. Relative IMU heading still regulates the turn toward the requested heading, so steering continues through gaps.
3. Lane loss for up to 0.3 s keeps full speed; speed then ramps linearly to zero at 1.0 s (about 11 cm of travel at speed 25). The robot then waits stopped and resumes after two consecutive lane frames. Three seconds without the lane, a heading beyond 10 degrees, unavailable/slow IMU, stale frames, or a delayed loop end the run as before.
4. The 8 px hard deadband is removed. The turn command changes at most 40 units per second and is rounded only when sent to the motors. An optional slowdown during large turns exists but is off, because only forward command 25 has been validated.

Logs add `filtered_error`, `lost_s`, the unrounded `turn`, and states `driving`, `coasting`, and `waiting`. `track_vision.py --detector dashes` keeps the old detector for comparison; `lane` is the default. Fifty-two simulated checks pass. Gains and coast timings remain provisional until physical trials. Rerun the four-trial suite with the same command and upload procedure.

### Run-improvement loop: record, replay, score, regress

Each run now produces data that improves the next one.

1. **Record.** Every camera frame is JPEG-encoded (quality 85, about 43 KB) on a worker thread and kept in memory; files are written to `frames/` only after the motors stop. If encoding falls behind, frames are dropped and counted, never delaying control. A 14-second course run is about 350 frames, roughly 15 MB on disk and 20-30 MB of RAM, so Pi memory size is not a constraint (`free -h` shows it). Lane position from quality-85 JPEGs is within 2 px of the original PNGs. `--record-every N` keeps every Nth frame; 0 disables recording. `reference_raw.png` stays lossless.
2. **Score.** After each run, `metrics.json` is saved and a one-line score is printed: percent of frames tracked, lost-lane episodes and longest gap, waiting time, near-row error RMS/peak/final, time until the robot stays within 10 px, turn jitter, turn reversals (weaving), and peak heading. `python run_metrics.py centering_results` prints a table across all runs; older runs are scored from `readings.csv`.
3. **Replay.** `python replay_run.py centering_results/run_X` runs recorded frames, IMU headings, and timestamps through the same `Controller` the robot uses, then prints recorded vs replayed scores. `--set KEY=VALUE` tries tuning changes against the recording; `--video` writes an annotated `replay.mp4`; every replay writes `events.jpg` with lost-lane and jump frames. Output goes to `replay_results/` (git-ignored). Replay is open loop: it shows what new code would measure and command for exactly what the camera saw, not where the robot would have gone, so gain changes still need physical runs.
4. **Regress.** `replay_run.py --mine` adds frames where the lane was lost or the fit jumped to `regression/labels.csv` as `candidate`, with a lane position interpolated from tracked neighbors within 0.3 s; `candidates.jpg` shows the proposed line in magenta. Reviewed labels become `confirmed` (or `rejected`), and `tests/test_regression_frames.py` requires every confirmed frame to keep tracking within 10 px near / 15 px far, or to stay untracked when no lane is measurable. The set starts with 14 visually checked frames: blurred powered failures, rotation blur, border-cut lanes, the carpet false-start frame, offsets, and four no-lane negatives.
5. **Tune without code edits.** `--set` on `quiz3_autonomous.py` and `quiz3_course.py` overrides tuning values (gains, smoothing, turn rate, coast timings). Speed, turn limit, abort heading, lost-lane limit, and sample-age limits are not overridable. Settings used are saved in `outcome.json`.

Per-suite workflow: run `python quiz3_course.py --execute` (optionally with `--set ...`), upload `centering_results/` with the usual commit/push, then on the development machine run `python replay_run.py centering_results/<new runs> --mine` and `python run_metrics.py centering_results`. Review candidates, confirm labels, fix any failing regression frames, and compare scores before the next suite. Expect about 60 MB of JPEGs per four-trial suite in git. Eighty-one simulated checks pass, including a recorded-then-replayed run that reproduces the live decisions.

### First recorded course suite (October 11, 2026)

Runs `run_20261011_212015_704644` (straight), `212215_684522` (left), `212513_828268` (right), and `212713_385597` (nudge). One earlier attempt (`211947_614675`) stopped at the reference check. Every frame was recorded with none dropped; the loop ran at 20-22 fps with 25-29 ms mean processing.

- The lane detector tracked 100% of frames in the straight, left, and right trials; every tracked frame across the suite was a correct fit.
- Operator notes: straight "pretty centered, a little movement to the right"; left "a little too much overcorrection to the right", and it ran about 9.5 ft before Enter; right "didn't correct enough to the left".
- Nudge: the push rotated the robot from -0.1 to -10.75 degrees in about 0.4 s. The lane stayed visible and correctly tracked (174 px right of reference) with the controller already at full right turn; the 10-degree heading limit then ended the run. The run was recoverable.
- Steering dead zone: across all four runs, 0.5 s windows with mean turn command 1.5-4.5 rotated the robot about +0.1 deg/s (noise); only commands of about 5 and above turned it. Most corrections fell in the dead zone, leaving steady 10-15 px offsets.
- Rotation vs offset: capture 005 shows 13.1 degrees of rotation moves the near-row lane -272 px (about 21 px per degree), so most of the nudge's 174 px image error was rotation. The left trial's ~3 cm slide read about 28 px of offset after removing rotation (roughly 9 px/cm, uncertain).
- Speed: about 14 cm/s at command 25 (13.5-15 cm/s across trials), not 11, so 5 ft takes about 11 s and the old 27.7 s time fallback was about 13 ft.
- Replay with the code that ran reproduced the recorded scores (tracked %, error RMS within 0.1 px, jitter). It also showed a fragility: when a nudge frame's confidence dipped under 0.5, the search window stayed put while the lane moved about 40 px/frame and outran the 64 px gate.

### Nudge recovery and dead-zone compensation

1. Lateral estimate: the IMU-measured rotation (`rotation_error_per_deg` = 0.0323, about 21 px per degree) is removed from the near-row image error, so rotation is not mistaken for sideways offset. The estimate sets the target heading (`position_gain` 60, max 6 degrees); IMU heading regulates the turn as before.
2. Lane out of view: the lateral estimate is held and the robot keeps driving at speed 25, steering on IMU heading toward the side where the lane was. It returns to normal tracking as soon as the lane is found; 3 s without the lane (about 42 cm) stops the run. The slow-to-stop-and-wait behavior is removed.
3. The search window for the next frame is shifted by the rotation since the lane was last seen, and `min_confidence` is 0.35 (17 correct frames in this suite fell between 0.3 and 0.5).
4. Dead-zone compensation: turn commands of at least 0.5 are mapped to 4 plus the remaining range, so small corrections actually rotate the robot (`turn_deadzone`, tunable; 0 disables).
5. Heading abort raised from 10 to 45 degrees as a runaway guard rather than a disturbance limit.
7. Steering limit raised from 10 to 20 (`--max-turn`, at most 20) without a separate calibration run, by decision to test it together with the other changes. Turn 10 gave only about 3-4 deg/s. The heading loop asks for more than 10 only when about 3+ degrees off, so normal tracking stays gentle (replay: at most 7 before the nudge) and recovery gets the extra authority (replay: ramps to 17 during the nudge). Staying below speed 25 should keep mixed mode from reversing the inner wheel. In the model, a 15-degree nudge was out of view 0.8 s instead of 2.1 s, and a 25-degree nudge recovered instead of stopping; the model's linear extrapolation of turn rate above 10 is unmeasured.
6. Course runs now stop automatically at an estimated 5 ft (one mat), integrating commanded speed at the measured 14 cm/s (`--estimated-speed-cm-s`); Enter still stops sooner and the time fallback remains. Each row logs `distance_cm` and `lateral_error`; the scorecard adds estimated distance, lateral RMS, and search time.

A rough closed-loop model from these measurements (dead zone 4.5, about 0.7 deg/s per turn unit above it, 14 cm/s) reproduced the observed steady offsets and the nudge abort with the old controller. With these changes it recovered an 11-degree nudge to within about 0.6 cm and a 15-degree nudge after about 2 s without the lane; a heading gain of 5 or position gain 80 added weaving. Replaying the suite with the new code tracks nudge frames 200-202 and searches with full right turn at frame 203 instead of aborting. Nudge frames 200-202 were added to the regression set. Gains remain provisional; steering authority (turn 10 is about 3-4 deg/s) limits how fast large nudges recover. Possible next steps if needed: a steering-direction asymmetry check (calibration suggested right turns respond more strongly) and a validated turn limit above 10.

### Second recorded course suite (October 11, 2026)

Runs `run_20261011_214721_248543` (straight), `214821_207942` (left), `214940_946079` (right), and `215102_496234` (nudge), with the nudge-recovery controller and turn limit 20. One earlier attempt (`214658_543603`) stopped at the reference check. No frames were dropped.

| Trial | Lateral RMS | Lane tracked | Operator note |
| --- | ---: | ---: | --- |
| Straight | 2.6 px (was ~10) | 100% | a little to the right but mostly straight |
| Left | 17.1 px (incl. start offset) | 99.1% | perfectly centered |
| Right | 31.0 px (incl. ~5 cm start offset) | 100% | a little to the right; should correct a little faster |
| Nudge | 9.3 px | 100% | corrected well with nudges |

- The nudge trial reached 12.1 degrees and a 273 px image error while tracking every frame, then finished within 0.6 px. It ended at the estimated-distance stop.
- Measured steering response (0.25 s windows): turn 4-6 gives under 1 deg/s, about 10 gives 6-8 deg/s, 12-20 about 9 deg/s, roughly 0.95 deg/s per unit above 4.5. Left and right were similar; turn 20 caused no reported problem.
- Right trial: heading followed its target closely, but a 5 cm offset requested only about 4.5 degrees, shrinking as it closed, so correction had a time constant of about 5 s. Raising position gain is the fix, not more steering.
- Distance estimate bug: it integrated time from each motor command to the next decision and missed the motor serial write (about one third of each 47 ms cycle). Straight/left/right ended by Enter at 10.6 s, consistent with 5 ft at 14 cm/s, but the estimate read about 100 cm; the nudge run's "152 cm" auto-stop was about 7 ft of real travel (15.9 s).
- "A little to the right" with 2.6 px lateral RMS means the robot held the captured reference; center the robot carefully when the reference is captured.
- Mining flagged left-trial frame 172: a correct fit at confidence 0.32, just under the 0.35 trust threshold. Confirmed as a regression label.

Changes: distance integrates between consecutive decision times (new test with 15 ms motor writes catches the old bug); `position_gain` 60 to 90 and `max_target_heading` 6 to 8 degrees. The model, updated with the measured steering response, shows about 1.5x faster settling (5 cm offset: 9.5 to 6.3 s) without the extra weaving of gain 120+. Replay of the right trial targets up to 7.2 degrees (was 4.9); the straight trial is unchanged.

Edge starts: the next left/right trials start half off the road at the edge, as graded ("recover from either half-off-road start within 5 feet"). That is about 15 cm, roughly 140 px of near-row image offset at ~9 px/cm, so the lane should stay in view, with the border-cut side measured from the dashes. Two changes support it:

- No search prior before the lane is first found. The robot is moved after the centered reference is captured, and a prior at the reference would reject a lane 140 px away (64 px gate) for the first 0.3 s.
- `max_target_heading` 8 to 12 degrees. In the model, a 15 cm start first reaches within 2 cm at about 3.7 ft with 8 degrees and about 3.0 ft with 12 (15-20 degrees adds little). The model has been optimistic, so the margin matters. Turning toward the lane moves it back across the image, so 12 degrees keeps it in view; small offsets and nudges never reach the limit.

A loop test shifts the centered image 150 px and requires tracking from the first frame with a full 12-degree approach; it fails under the old reference prior. No half-off-road frames exist yet, so detection from the edge is unverified until this suite is recorded.

### No centered reference: calibrated center and image heading

The graded test does not allow capturing a centered reference first. The reference capture supplied two things: where the lane appears when centered, and the IMU zero for "pointing along the track". A one-degree zero error alone holds the robot about 1.5 cm off center.

1. `lane_center.json` stores the centered-and-straight lane position (near row 0.639, far row 0.5927): the median over the eight operator-centered references from the two recorded suites, whose spread was about 7 px. The camera is fixed, so this is a robot property. `python lane_center.py IMAGE...` recomputes it from centered images if the mount moves.
2. Heading from the image: a sideways offset moves the far row 0.60 as much as the near row (captures 002/003), while rotation moves it 0.91 as much (capture 005), so far minus 0.6 x near depends only on rotation. Across all eight recorded runs, image heading tracked the IMU with slope 0.83-1.28 and 0.2-0.7 degree per-frame noise, including the nudge's -2 to +12 degree swing. Per-run offsets of -0.1 to +1.0 degree were the by-eye straightness of each old reference. On the labeled captures it matches the IMU within 0.5 degree, and the rotated capture reads +1.3 px of lateral offset (rotation correctly not counted as offset).
3. Start-up: after `RUN`, the robot holds still for ten frames (~0.5 s), requires the lane in at least half of them, and sets the IMU zero so the relative heading starts at the median image heading. It prints the start heading and lateral offset; without the lane it refuses to start. `start_raw.png`, the start measurements, and the calibrated reference are saved with the run; replay uses `reference.json` when there is no `reference_raw.png`.

Trial prompts now describe placement only (centered, or half off at the left/right edge). The heading reference is fixed at start-up; if IMU drift becomes visible on longer runs, image heading (now well characterized) can slowly correct it. A motor-free end-to-end test runs `main` with fake camera/IMU and checks for no reference prompt and complete outputs.

### Third suite: edge starts without a reference (October 11, 2026)

Runs `run_20261011_220555_668493` (straight), `220712_368437` (left edge), `220819_453756` (right edge), `220938_068460` (nudge), in a different room; all four stopped at the estimated 5 ft. Start-up found the lane in 10/10 frames from both edges (+174 px and -160 px lateral), and the straight trial held within about 5 px, so the fixed `lane_center.json` calibration carried over.

- Notes: left "corrected too far to the right"; right "pretty good, not enough to the right" (ended left of center); nudge "started correcting but 5 ft too short".
- Cause of both overshoots: while held at about 12 degrees, the IMU heading drifted 2-4.5 degrees away from the image heading (left trial ended IMU 7.5 vs image 2.9). The rotation-removed lateral estimate inherited the error and still read "not there yet" at center. Image-only lateral matched the operator (left ended -40 px, right +30 px) and closed at a physically consistent ~30 px/s.
- Fix: whenever the lane is trusted, the IMU heading is pulled toward the image heading with weight 0.1 per frame (~0.5 s; innovation clamped to 5 degrees), held while the lane is unseen (`heading_correction`, 0 disables). Replay of the edge trials then eases the target from 12 degrees starting about 4.6 s in and reaches center nearly straight, instead of holding 12 degrees until 6.6 s.
- Detector: another robot parked down the track (and a shoe during the nudge) sat 25-45 px beside the lane at the far rows, inside the 64 px prior gate, and wrecked the fit (right trial 90% tracked). Row continuity now applies with a prior too, follows the lane's own lean once five rows are found (median pairwise slope, 0.15 px/row allowance), and the robust refit runs twice. Replay: right trial 100% tracked, nudge 95%; older frames still agree with the dash fit within 3.4 px; about 2.2 ms per frame on the Mac. Mining added 42 labels from these runs (35 confirmed, 4 edge-sliver negatives, 3 corrected where interpolation across the nudge was off by ~20 px); 60 confirmed labels pass.

### One program for every test, 15 ft legs

The exam runs all tests in one program: the robot is moved between tests and must know where to go. `quiz3_autonomous.py` (and `quiz3_course.py`, now an alias) runs a session:

1. Wait, stopped: once the lane has been seen steadily for 1.5 s, start. After a leg, the robot must first be moved, either picked up (lane unseen 0.5 s) or slid (lane shifts more than 40 px), so it never restarts on its own where it stopped. Enter starts at once; `q` + Enter quits.
2. Leg: hold still ~0.5 s to measure heading and offset from the image (sets the IMU zero), then drive until an estimated 15 ft (`--leg-ft`, at most 20; time fallback 2x nominal, capped at 90 s). Enter ends the leg.
3. Lane not found for 3 s, rotation past 45 degrees, or no lane at start ends only that leg: motors stop, the leg is saved, and the robot waits to be placed again. Hardware/IMU/timing faults still end the program.

Each leg is saved as its own `run_*` folder (trial `leg N`), so replay, mining, and the scorecard work unchanged. To keep pushes small, every 3rd frame is recorded by default (`--record-every 3`, about 10 MB per 15 ft leg; `--record-every 1` keeps all). Replay then steps the controller only on recorded frames, so decisions approximate the live ones more loosely than with every frame. No prompts appear between legs. A 15 ft leg is about 33 s and roughly 30 MB of recorded frames. Session tests cover first placement, no restart until moved, pick-up and slide restarts, Enter/q, two legs from different starts in one session, and a lane fault that ends one leg but not the session.

### First one-program session (October 11, 2026)

Legs `run_20261011_222608_014418`, `222712_568548`, `222851_395272` (each started as leg 1). Operator: very happy overall; corrected successfully for the most part, a little off here and there.

- Edge starts are clean: both right-edge legs went from about -180 px to within 10 px by about 117 cm (3.8 ft) with no overshoot; image heading correction held the IMU's 3-4 degree drift out. Nudges mid-leg recovered.
- Every leg ended on a fault, and none was a control failure: leg 1 when a person's leg blocked the camera; legs 2 and 3 at the end of the track (another robot parked there). Leg 3 kept searching toward that robot for about 5 s because occasional detections of the remaining lane stub reset the 3 s limit; leg 2 searched until rotation passed 45 degrees.
- End-of-lane stop: if the lane is seen only in the near rows (no point above image row 0.68) with no confident full-lane frame for 0.5 s, the leg ends with "End of lane ahead" (recoverable: stop, save, wait). Normal tracking reaches row 0.67 or higher in 99% of frames. Replay triggers it on all three new legs (17.0, 28.0, 25.8 s) and on none of the sixteen earlier runs, including nudges and edge starts.
- Detector: the previous session's tight row-to-row chaining broke two of the eight centered reference images (a wrong bottom row locked out the real lane). Row chaining is replaced by a consensus line: every row offers candidate centers, lines through near/far candidate pairs are scored by how many candidates lie within 2% of the width, and the best-supported line wins (ties to wider runs). All eight references track again (0.71-0.86) and were added to the regression set (68 confirmed labels). Agreement with the dash fit improved (max 2.4 px), it runs about 1.3 ms per frame, and replay tracking improved on every run (hardest new leg 66% to 79%).
- "A little off": the image shows a slight lens bend (near-half lane slope 0.03-0.15 below the far half), and in one stretch the image heading read about 4 degrees while the lateral estimate did not change, i.e. a few-degree heading bias worth roughly 2-3 cm. A quadratic fit did not improve physical consistency, so the straight-line fit stays. Plain straight legs without nudges would isolate this.
- One frame previously labeled as a no-lane sliver (nudge run, frame 222) actually shows the dashes; relabeled.

### Option: never stop for a lost lane

`--keep-searching` disables the 3 s "Lane not found" limit, so a leg runs past the end of a short track (or through any lane loss) on IMU heading until the estimated distance, Enter, or 45 degrees. On the ~3 m track in the latest room, both legs stopped after the lane ended at the wall, with nudges in the last half meter. Off by default; the operator must stop the robot by Enter or the physical stop.

### End-of-lane stop removed

In a short-track room every leg ended "End of lane ahead" about 0.5 m before the mat met the wall (leg 1 after ~147 cm; restarts nearby after 8-20 cm). By request the stop is removed: a leg now ends at the estimated distance, Enter, 3 s without the lane, or 45 degrees of rotation. Keep the end of the track clear, or press Enter, since a parked robot at the end is no longer avoided.

### Dim-room failure and lane color fix (October 11, 2026)

Legs `run_20261011_223932_140284` and `223959_429264` in another room performed badly: confidence 0.3-0.6 from the first frame, lane often lost, and a wrong fit swung the robot 12 degrees away. It was not the consensus detector (on these frames older detector versions usable 5% and 14%, consensus 25%). The lane mask was: in this room the green mats are dim (brightness 119-130) and washed out, passing the "dark, unsaturated" test, so lane and mats merged; the lane itself is more saturated (about 115) than elsewhere.

Measured across the four rooms: mats hue 42-57 at brightness 112-163; lane brightness 55-76, hue 84-106 (greenish in the first room). The mask now excludes green-hued pixels only when bright enough to be mat (hue 30-75, saturation over 40, brightness 90+), and allows lane saturation up to 135. A dark-blue rule was tried and dropped: the blue robot parked at the track end then read as lane and suppressed the end-of-lane stop.

Usable frames: new room 25% to 93% (median confidence 0.82); previous session 77% to 92%; all older footage stays at 100% (confidence 0.91). Dash-fit agreement max 2.5 px. Replay: the two bad legs track 89% and 100%; end-of-lane triggers at all three track ends and nowhere else. Eight new-room frames added to the regression set (76 labels).

### Touchscreen app and the "not moving" session

Five legs (`run_20261011_2255*`-`2257*`, motors enabled) logged up to 23 s of "driving" (speed 25, turn 7) while the camera image and IMU heading never changed: first and last frames identical. The robot was not moving; the software kept commanding it, which looked like a freeze, and three legs ended by an outside interrupt. Most likely a drained battery (speed 25 is near the minimum that moves the robot) or motor power off.

`lane_app.py` is the simple touchscreen program, by request: the screen shows the camera with the detected lane (green) and calibrated center (cyan). START measures position (~0.5 s still), then centers on the lane and drives straight until STOP, the leg distance (15 ft), or a fault; it then waits for the next START. QUIT exits. There is no automatic start on placement. While driving, if the commanded robot shows no image or heading change for 2 s, the screen shows "NOT MOVING - check motor power / battery". Runs save like legs (replay and metrics unchanged). `quiz3_autonomous.py` keeps the terminal session mode. Tests drive START/STOP/QUIT with a fake screen, a no-lane start, the no-motion warning, and `main` with fake hardware.

### First touchscreen-app session (October 11, 2026, motors working)

Legs `run_20261011_230521_064421` (center), `230602_508233`, `230634_327418`, `230726_473761` (left edge, +177 to +201 px). Operator: worked really well. (`230315_491877` earlier shows the not-moving signature: before the motor fix.)

- Center start: nudges at ~3 ft and ~8.5 ft recovered; full 15 ft, finished centered.
- Left-edge starts: within 15 px of center at 3.6 ft (from +177 px) and 5.7-5.8 ft (from about +190-200 px), about 3 cm off at 4.6-4.8 ft; the 5 ft requirement is met only for the smaller offset. The slow part is the final proportional approach.
- Legs 2 and 4 stopped "Lane not found" at about 10-11 ft because the mats ended (floor and a chair ahead), not a fault; they started farther down the track than legs 1 and 3.
- Model check with the measured steering response: position gain 110 or 130 reaches center about 0.3-0.5 ft sooner (model, optimistic by ~1.35x versus these runs) with somewhat more small corrections. Not made default; try `--set position_gain=110` on edge starts and compare scores.

### No distance limit in the touchscreen app

By request, `lane_app.py` runs have no distance stop by default: a run ends on STOP, 3 s without the lane (for example past the end of the mats; `--keep-searching` disables this), 45 degrees of rotation, or a 10-minute safety time limit. `--leg-ft` (up to 100 ft) still sets a limit; `quiz3_autonomous.py` keeps its 15 ft default with the cap raised to 100 ft.

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
