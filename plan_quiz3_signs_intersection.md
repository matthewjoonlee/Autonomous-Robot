# Quiz 3 plan: intersection detection and sign reading

Sources: Week 11 (pages 20-24, the Quiz 3 requirements and judge forms), Week 10 (location service design), Week 9 (sound). Copies are in `course-materials/Class Presentations/`.

Note: the earlier [plan.md](plan.md) is also titled "Quiz 3", but it covers lane centering. Week 11 redefines Quiz 3 as the two tests below. Lane centering is still needed: it carries the robot through the intersection test. It is also a separate final-demo station.

## 0. Confirmed facts (November 9, 2026)

- **Date and room:** Quiz 3 is Monday, November 16, 2026, in class. Collect test images in the classroom lighting.
- **Mats:** the intersection is either the four-way mat (#6) or the right-turn mat.
  - Both have a road branching off to the right, which is the side the side camera faces.
  - The right-turn mat is probably #1 or #5: straight through, with a right branch. #9 is a right turn only, with no straight path out. Confirm which one.
- **Zone lines:** the mats have no zone lines. We define them ourselves (section 3).
- **Reference point:** zones are judged by the **center of the robot**.
- **Starting positions:** the meaning of "two different road starting positions" is unclear. Design for any start before the pause zone, including a sideways offset, which lane centering already handles.
- **Signs:** you have the physical signs. They sit at the side of the road.
- **Side camera:**
  - It points out the right side, and the mount is final.
  - Its main job is telling black road from green grass, so blur is acceptable. The lens can be refocused if the yellow cross dashes need to be sharper.

## 1. Quiz 3 requirements (extracted)

The PS5 controller selects which quiz mode is active. 100 points total.

### A. Intersection detection: 50 points (two runs × 25)

- The robot drives forward on its own toward an intersection. It does this twice, from two different road starting positions.
- It drives straight through the intersection. Judges score whether it identifies the turn zone; it does not have to turn.
- For each of the six phases below, it must report its location and announce the phase aloud. It must also show the phase on the display.
- It should pause briefly at each phase change so the judges can score.
- The judges use the same physical zone boundaries for both runs.

| Phase (in driving order) | Judge's description |
| --- | --- |
| Read the sign | Sign-reading approach zone |
| Pause zone | Pause zone before entry |
| Entering intersection | Entry into the intersection |
| Turn zone | Where a turn would begin |
| Exiting intersection | Intersection exit zone |
| Continue travel | Travel beyond the intersection |

Scoring per run:

| Item | Points |
| --- | --- |
| Location, per phase | 2 for the correct zone; 1 if early or late into an adjacent zone; 0 if missing or wrong |
| Audio, per phase | 1 for a correct, understandable announcement |
| Display, per phase | 1 for a readable phase label |
| Operating behavior | 1 for automatic travel plus judging pauses; 0.5 if only one of the two |
| Run total | 6 × 4 + 1 = 25 |

Audio and display are scored separately from location accuracy. A late announcement still earns the audio point, but the location may score only 1.

### B. Sign reading: 50 points

- The robot stays stationary. Staying stationary earns no points.
- The judges present each of the four signs once, in shuffled order: **Left, Right, Straight, Pause**. Sign placement and lighting stay the same throughout. The observation window is about 5 seconds after each sign is positioned.
- The mode is active without any trigger from the location service. It must keep recognizing signs after the first one.

| Item | Points |
| --- | --- |
| Mode shown on screen as "Sign Reading" | 2 (1 if unclear or intermittent) |
| Bounding box, per sign | 4 for a stable box that closely encloses the sign; 3 for a minor size or stability issue; 2 for substantial clipping, oversizing or instability; 1 for occasional overlap; 0 if missing or around the wrong object |
| Display label, per sign | 4 if correct and consistent within the window; 3 with a minor delay or brief error; 2 if eventually correct but repeatedly inconsistent; 1 if only briefly correct; 0 if wrong or missing |
| Audio label, per sign | Same scale as the display label |
| Total | 2 + 4 × 12 = 50 |

### Architecture rules from Weeks 9-11 (not scored directly, but required design)

- **Location service** reports evidence: where the robot is. It does not classify signs, decide to pause or turn, command motors, or run the workflow. Its cues are `sign_read_ready`, `pause_point_ready` and `turn_point_ready`. Each should hold over a small range, not a single pixel. It also reports the continuous zones (continue travel, entering, exiting) all the time.
- **Sign detection** reports what the sign says, as a box plus a label.
- **The main controller** owns camera capture: capture, resize, enhance, then write the latest frames with a frame id and timestamp. It owns all decisions, motors, the display and sound.
- **Workers** run in separate Python processes. They use the latest-frame pattern (freshness beats completeness, no queues of stale work). They return small annotation structures (rectangle/ROI, line, point, text, color, camera id) and the controller draws them. Workers never command motors.
- **Sound** must not block the control loop. Every sound that can start must be stoppable and have adjustable volume.
- **Engineering log**: record which approach was chosen and why, the inputs it needs, how an image becomes a box and a label, and what failed, what changed and what was retested.

## 2. What we already have

| Need | Existing piece | Gap |
| --- | --- | --- |
| Drive straight on the lane | `quiz3_autonomous.Controller`, `track_vision.detect_lane`, IMU heading hold, replay/regression tools | Untested through an intersection, where the cross road removes the green edges and adds horizontal dashes |
| Cameras | Front camera setup in `test_display.py` and `calibrate_track.py`; side camera in `capture_side_camera.py` | No single capture owner, no shared frames, side camera mount and focus not settled (the 2026-11-09 shots are blurry) |
| Display | `RobotDisplay` with `set_frame`, `set_status` and buttons; `lane_app.py` renders the overlays | No annotation-structure layer and no mode banner |
| Sound | `USB_SoundController` (queued worker thread) | Each TTS call runs `text2wave` first, which delays it, and queued phrases can back up and play late |
| PS5 | `PS5_Controller.check_controls()` with Cross, Circle, Triangle, Square, D-pad and so on | Not yet used together with autonomy |
| Signs | Supplied artwork: `left_arrow.jpg`, `right_arrow.jpg`, `Up_Arrow.jpg`, `stop.jpg` | No photos of the real signs from the robot's cameras |

## 3. Our zone definitions (we draw these)

We define each zone by the **robot center's** distance `d` from the center of the cross road, where `d` is positive before the center. Every boundary sits on a feature that is printed on the mat, so it is easy to show the judges and easy for the cameras to detect.

The four-way mat measures 1080 × 1800 px. Taking it as about 5 ft long gives 1 px ≈ 1/30 in, which makes the road about 12.2 in wide. That matches the measured 32 cm, so the scale holds. The cross road runs from +6.1 in to -6.1 in, and the corner curves start about 12 in before the center.

| Phase | Robot center range | Feature marking the end of the zone | Main sensor |
| --- | --- | --- | --- |
| Continue travel (approach) | d > +24 in (61 cm) | A tape mark 12 in before the curves start | Front camera |
| Read the sign | +24 to +12 in (61 to 30 cm) | The corner curves start | Front camera |
| Pause zone | +12 to +6 in (30 to 15 cm) | The near edge of the cross road | Side camera: the green edge curves away |
| Entering intersection | +6 to +1.5 in (15 to 4 cm) | Just before the cross dashes | Side camera: black road, no dashes yet |
| Turn zone | +1.5 to -1.5 in (±4 cm) | Center over the cross dashes | Side camera: yellow dashes at the center column |
| Exiting intersection | -1.5 to -6 in (-4 to -15 cm) | The far edge of the cross road | Side camera: black road after the dashes |
| Continue travel | d < -6 in (-15 cm) | (no end) | Side camera: green again; front camera: plain lane |

To do:

- **Check the numbers on the real mats** with a tape measure: the cross-road edges, the curve start and the dash line.
- **Make the zones visible for the judges.** Put removable tape ticks at the mat edge, or bring a printed copy of this table with a diagram.
- **Right-turn mat:** the right side matches the four-way mat. Only the left side differs, and the plan does not use the left side for location.

## 4. Software design

New files; existing lane-centering files are left unchanged except where noted.

```
autonomy_app.py      main controller: PS5 mode select, capture, decisions, motors, display, sound
frame_hub.py         shared-memory latest frames (front + side) with frame_id, timestamp
annotations.py       Rect / Line / Point / Text dataclasses (+ camera id, color); draw()
location_service.py  worker process: frames -> LocationResult
sign_service.py      worker process: frames -> SignResult
announcer.py         non-blocking speech: pre-rendered WAVs, latest-wins, stop/volume
```

Interfaces (keep these stable even if the detection method changes):

```python
LocationResult(frame_id, timestamp, zone,            # one of the 7 zones above or "unknown"
               distance_cm, confidence,              # estimated distance to intersection center
               sign_read_ready, pause_point_ready, turn_point_ready,
               evidence: dict, annotations: list)

SignResult(frame_id, timestamp, label,               # LEFT/RIGHT/STRAIGHT/PAUSE/UNKNOWN/NONE
           box, score, scores: dict, annotations: list)
```

- **Latest-only results:** each worker reads the newest frame id from shared memory and skips any frames in between. Results go into a queue of size 1; the worker replaces an unread result instead of adding to it.
- **Controller hints:** the controller sends hints to the workers (mode, commanded speed, IMU heading). Hints are evidence only and do not give the workers any authority.
- **Announcer:** at startup, render every fixed phrase to a WAV file once (six phases, four signs, "Unknown sign", the mode names). At runtime, playing a phrase means playing its WAV, with no `text2wave` delay. A new status phrase replaces any phrase still waiting in the queue, so announcements never play late.
- **PS5 mapping (proposed):**
  - Square selects Intersection Detection.
  - Triangle selects Sign Reading.
  - Cross starts a run.
  - Circle stops motors immediately in every mode.
  - The touchscreen has STOP and QUIT as backups.
- **Display:** a permanent top banner shows the mode ("SIGN READING" or "INTERSECTION DETECTION") and the current phase or sign label in large text. The camera panels show the worker annotations.

## 5. Sign reading approach: OpenCV first

Recommendation: rules-based OpenCV. The four signs are high-contrast, known artwork: a white panel with a black border, black arrows, and a red octagon. The quiz is stationary, with consistent placement and lighting. This approach needs no labeled dataset or training setup, runs comfortably on the Pi, and every intermediate step can be shown on screen.

Switch to a trained detector only if the comparison test in step 6 shows the rules failing on real images.

Pipeline, run on every frame in the worker:

1. **ROI:** the whole front frame for the quiz. When driving, narrow it to the left side ahead.
2. **Find the sign:** threshold for bright, low-saturation pixels, find contours, and approximate each one with `approxPolyDP`. Keep convex quadrilaterals above a minimum area, within a plausible aspect ratio, and with a dark border or inner content.
3. **Box:** report the bounding rectangle of the outer quad, with a little padding so the box encloses the whole sign. The box score rewards a stable fit that closely encloses the sign.
4. **Normalize:** warp the quad to a canonical crop so the full arrowhead stays in view (Week 11 page 7).
5. **Classify:**
   - If the red fraction inside the crop is high, the label is PAUSE.
   - Otherwise, binarize the black shape and run normalized template matching against the supplied left, right and up artwork. Also apply a shape check: the arrowhead side is the wider end of the black blob, and a STRAIGHT arrow is taller than it is wide.
   - Reject the result as UNKNOWN unless the best score clears a threshold and beats the second-best by a margin.
6. **Temporal filter:** show and announce a label only when at least 3 of the last 5 results agree. Hold the box and label until a different stable label appears or the sign has been missing for about 1 s. Announce once per new stable label, and announce again if the sign disappears for about 1 s and comes back. This keeps recognition going after the first sign.

## 6. Location service approach: combined evidence

The robot has no wheel encoders. Each zone is assigned to the camera that can actually see its feature at the moment the robot's center reaches it.

1. **Front camera: approach, read the sign, and the start of the pause zone.**
   - The camera is mounted rigidly, so each image row corresponds to a fixed distance on the floor. Calibrate this row-to-distance mapping from captures at known distances.
   - Detect the near edge of the cross road: the row where the green on both sides of the lane ends.
   - The robot center's distance `d` is the edge distance plus 6 in, minus how far the front camera sits ahead of the robot center.
   - This works from any starting position.
   - It stops working when the near edge goes under the robot's nose, around the pause zone.
2. **Side camera: pause zone through continue travel.**
   - The camera looks at the ground to the right of the robot.
   - Calibrate one image column: the column that lines up with the robot center. Place the robot with its center exactly on a tape line and capture.
   - At that column, classify the ground band nearest the road as one of four things:
     - green with a straight edge: before the curve;
     - green receding: the curve, so pause zone;
     - black: cross road;
     - black with yellow dashes: turn zone.
   - Blur is fine for telling green from black. The dashes may need the focus adjusted.
3. **Carry-over between sightings.** Update `d` with elapsed time × speed. Estimate the speed from how fast the front-camera edge moved through the image while it was visible. The side-camera events replace the estimate whenever they fire.
4. **Output:** the zone for the current `d`, with about 1 cm of hysteresis, plus a confidence. Report the cues over their whole band.

The controller keeps the phases in order: continue, read, pause, entering, turn, exiting, continue. On each new zone it stops, shows and announces the phase, waits about 1.5 s, and continues. This earns the judging-pause point, and the next measurement is taken from a sharp, stationary frame. If a zone is skipped, the controller still announces it at once: that is worth 1 location point plus the audio and display points, against 0 for missing it.

Through the intersection, steering relies on IMU heading hold. The lane detector is only trusted when there is green on both sides.

## 7. Implementation phases

### Schedule to Monday, November 16

| Day | Claude (code) | You (on the robot) |
| --- | --- | --- |
| Mon 11/9 to Tue 11/10 | Phase 1 infrastructure. `capture_signs.py`, `capture_intersection.py`. First sign detector run on the supplied artwork | Measure the robot and the mats. Capture sign photos and photos at tape-measured positions on both mats, with both cameras, in the classroom |
| Wed 11/11 | Tune the sign detector on your photos. Sign Reading mode in the app | Mock sign quiz, scored with judge form 2 |
| Thu 11/12 | Front and side location detectors, evaluated offline on the labeled captures. Location worker | Re-capture anything the detectors need |
| Fri 11/13 | Intersection mode: lane centering plus heading hold, with phase stops | First powered runs at low speed, recorded |
| Sat 11/14 to Sun 11/15 | Fix issues from the replays | Both mats, several starts, mock runs scored with judge form 1. Freeze the code Sunday night |

Priority if time runs short:

1. Sign reading (stationary, 50 points).
2. Intersection-run announcements, display and pauses for every phase. These are 2 of the 4 points per phase and do not depend on location accuracy.
3. Location accuracy.

Each phase ends with a check you can run.

### Phase 0: measurements

- Answer the open questions in section 9.
- Measure the robot's length, and the positions of the front and side cameras relative to the robot center.

### Phase 1: shared infrastructure (motor-free)

- Write `annotations.py`, `frame_hub.py` and `announcer.py`, and an `autonomy_app.py` skeleton with PS5 mode selection, the mode banner and a dummy worker that returns annotations.
- **Check:**
  - Both camera panels update.
  - Square and Triangle switch the banner.
  - The dummy annotations draw on the correct panel.
  - Phrases play with no audible delay, and a burst of phrases plays only the latest.
  - The main-loop rate stays steady while a worker deliberately burns CPU.

### Phase 2: sign reading (stationary, motor-free)

1. Write `capture_signs.py`, based on `capture_side_camera.py`. Collect front-camera photos of each real sign, using the planned quiz placement plus several distances, angles and lighting conditions, and some frames with no sign. Store them in `sign_captures/`.
2. Run an offline experiment on the saved images. Save the ROI, the candidate crop, the prepared crop and the match scores for each image. Report a confusion matrix and the no-sign result.
3. Turn the experiment into `sign_service.py`, add regression tests on the saved images, and connect it to Sign Reading mode.
4. **Check:** score a mock quiz with the judge form: four shuffled signs, a 5 s window each. Target 50/50 in the quiz location and lighting.

### Phase 3: location service (offline first)

1. Write `capture_intersection.py`: with motors off, push the robot by hand. At each tape-measured distance from the intersection center, capture front + side frames labeled with the distance. Also take recorded slow drives through the intersection, with both cameras recorded.
2. Fit the row-to-distance calibration. Build the near-edge, cross-dash and side-camera detectors. Evaluate the per-frame zone accuracy against the labeled distances.
3. Write `location_service.py` with annotations: the ROI boxes, the detected edge and dash lines, and a text note with the zone and distance.
4. **Check:** on the labeled captures, every zone is correct or at most one zone early or late, and none is skipped.

### Phase 4: intersection run (powered, low speed)

1. Add Intersection Detection mode to `autonomy_app.py`:
   - lane centering on the straight sections, heading hold through the intersection;
   - stop, announce, display and pause for about 1.5 s on each phase change;
   - stop at the end of the mat or when the lane is lost.
2. Extend `run_recording.py` to record the side frames, so `replay_run.py` can replay the location service.
3. **Check:** run from both starting positions, score with judge form 1, and repeat until both runs score 25/25 consistently. Test in the quiz room.

### Phase 5: rehearsal and engineering log

- Run the full quiz back to back: mode switch, two intersection runs, four signs.
- Log the approach choices, the evidence, the failures and the retests (Week 11 page 26).

**Later, for the final demo (not Quiz 3):**

- turning, using the side camera to judge when a turn is complete;
- pausing at the Pause sign;
- exit detection;
- the manual-course ambient sound routine from Week 9: at least 5 sounds, chosen at random every 1-5 s, toggled from the PS5.

The `LocationResult` and `SignResult` interfaces stay as they are, so this work adds behavior to the controller only.

## 8. Risks

- **Front camera blind zone:** the turn and exit zones may be out of the front view entirely. This is why the plan uses side-camera confirmation plus carry-over based on speed.
- **Lane detection in the intersection:** the cross road removes the green on both sides and the cross dashes can pull the fit. Rely on heading hold in that region.
- **Lighting:** the lobby sun and dark spots already caused lane failures. Collect sign and intersection captures in the quiz room.
- **Sign box stability:** a box that flickers loses points. The temporal filter and the hold time address this.
- **Late audio:** solved by pre-rendered phrases and latest-wins.

## 9. Still open

1. Which right-turn mat: #1/#5 (straight with a right branch) or #9 (right turn only)? On #9, "continue travel" would have to be reported where the road ends.
2. The starting positions: ask in class if you can. Until then the plan covers any start before the pause zone, plus sideways offsets.
3. Measure the robot: its length, and how far ahead of the robot center the front camera and the side camera sit.
4. The signs:
   - Which side of the road are they on, left or right?
   - What are their size and height?
   - In the stationary sign test, where will the judges hold them relative to the robot?
