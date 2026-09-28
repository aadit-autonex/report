# Welding Detection Project — Work Done

## 1. Capture scripts for CCTV footage when welding is detected

- **Approach:**
  - Person detection
  - HSV mask that checked for (white) bright pixels
  - Temporal methods that checked for positive detections in 2-3 out of 5 frames
- **What it recorded:**
  - A burst of frames for 5 seconds
  - Raw/annotated frames for 3 mins @ 1 frame per 5 seconds (this was done because we wanted to check in how many frames the welding sparks were going out of a certain distance)
- **Result:**
  - This created a lot of false positives, but I also got around 2-3k frames that had welding in them for testing different methods on.
  - The person detection etc. was working most of the time, but welding triggers were off and giving a lot of FPs.

---

## 2. Spark-based welding detection on real CCTV footage from Shirval

- Ran the same spark detection methods on some frames:
  - Find the person using the DETR model
  - Create an ROI around them
  - Run only the HSV masks (removed temporal etc.)
- **Result:** this led to a lot of FPs (more than the ones I had with temporal).

[weld-test img]()

---

## 3. HSV values for welding vs. non-welding frames

- I wanted to see how the HSV values differed for all the welders vs non-welders.
- Annotated **311 crops** of humans as detected by the DETR model, in frames from **3 different cameras**.
  - DETR would detect, then add a 50% padding to the image so that the welding part can also be included in the frames.
- **Found out that:**
  - Blue colour dominates more than white.
  - Finding welding based on bright pixels is wrong, because even the pictures that don't have welding still have the brightest pixels somewhere in the ROI.
  - highly

---

## 4. Scraped feeds from all available cameras

- Set up scripts for recording frames @ 1 FPS from all the 60 cameras.
- Tested different approaches using DeepStream etc.; used some NVIDIA libs in the end that would ingest all the frames from the **57 cameras** that we have deployed at Shirval.
- Some of the cameras were not available on the RTSP stream and had to be recorded using the NVR:
  - Figured out the IPs that came under the specific NVR.
  - Set up the fallback for all the cameras that weren't available on RTSP to record from the NVR.
- Frames were recorded from the **substream at 640x360**.

---

## 5. (Substream) Tuned HSV thresholds based on the HSV analysis

- Tried tuning based on HSV results:
  - Tuning for **blue** basically made it detect the guy with the blue shirt.
  - Same thing for **yellow-orange**: it finds yellow hard hats.

---

## 6. Welding detection experiments on 1 FPS recordings

- Temporal differences go wrong in this, because @ 1 FPS, if I try to find welding with 60% confidence in 2-3 out of 5 past frames, then the recording doesn't exactly have the frames in which welding actually happens.

---

## 7. Need to prioritise recall for welding detection

- Accepting some false positives during initial experimentation.
- I tried increasing the thresholds, but it started missing all the welding frames (again, I think it's mainly due to the data being from the substream).

---

## 8. Welding timer

- Tiled inference + ByteTrack, with a bb box around them and a timer.
- Same HSV masks (top 0.1 percentile brightness).
- **Result:** very bad detection.
  - The tracker fails to track the person, and the welding detection was already not working, so it was tracking the wrong people.

---

## 9. Tried the opposite way

- HSV mask first, then find the person around it.
- **Result:** too many distractions, couldn't detect the right things.

---

# Experiments Log

Each entry: what I expected, what happened, what changed, and what to do next.
Read this before starting a new detector experiment.

## Test setup (applies to everything below unless noted)

- **Hardware:** Jetson Orin on site. Nothing was measured on the dev laptop.
- **Images:**
  - Mostly **substream** snapshots (low res, heavy compression), sampled from recordings/scrapes, often one still frame at a time.
  - Far-away welders are only a few pixels in these frames, and compression smears small sparks.
  - **Not yet run on mainstream (full-res) frames.** That could change most of the results below.
- **Person model:** `./rtdetr-l.pt` (COCO, not fine-tuned).
- **Thresholds:** all hand-picked, none tuned on labelled footage from these cameras. The only labelled set so far is about 768 welding + 38 sparks.
- **Cameras:** some are always unreachable and the clock is not NTP-synced. Neither affected the results, but both show up in the logs.

## Results

1. **Low-FP defaults (temporal required, 3-of-5 confirm, threshold 0.6) give clean detections**
   - **Result:** missed almost **all** welding.
   - **Gap:** Large. Precision was the wrong target.
   - **Change:** favour recall — one frame of static evidence can fire, temporal is off by default (`--temporal`).

2. **Temporal masks (frame diff + optical flow) make it more robust**
   - **Result:** still missed welds. They also **can't run on still images**, which removes over half the checks.
   - **Gap:** Large.
   - **Change:** per-frame decision by default. Use bursts of consecutive frames when temporal is needed.

3. **Glow = saturation filter + 12px min blob + morph open**
   - **Result:** erased real (small, distant) arcs.
   - **Gap:** Large.
   - **Change:** glow = top 0.01% of V, white or blue, no saturation test, no size floor.

4. **Blue hue marks an arc**
   - **Result:** picked up blue shirts, and tracked the guy in blue every time.
   - **Gap:** Large.
   - **Change:** blue only counts when it is also extremely bright. Don't take blue from the HSV profile.

5. **Orange/yellow hue marks sparks**
   - **Result:** yellow hard hats triggered it.
   - **Gap:** Medium.
   - **Change:** sparks/cutting = orange only (hue 5-20).

6. **Person-gated ROI cuts noise**
   - **Result:** cut noise, but crouched, occluded or far welders get no box, so no ROI and a missed weld.
   - **Gap:** Large. This is the main recall loss.
   - **Change:** bigger ROI (50% margin), person conf 0.45, tiled inference for small people, `tiered` mode (person check last, only lowers the score). Never go back to a hard person gate.

7. **Full-frame masks avoid the missed-person problem**
   - **Result:** too noisy: skylights, lamps, reflections.
   - **Gap:** Large.
   - **Change:** Tier 0 static-brightness map per camera, and candidate regions before scoring.

8. **Fixed HSV thresholds (e.g. V > 240) work everywhere**
   - **Result:** break with exposure and camera changes.
   - **Gap:** Medium.
   - **Change:** dynamic, e.g. the brightest 0.1% of pixels.

9. **One method for welding and cutting**
   - **Result:** they look different: welding is a glowing blob, cutting is small motion-blurred orange streaks.
   - **Gap:** Medium.
   - **Change:** plan: separate scorers.

10. **Grouping the HSV profile by the detector's own score shows separation**
    - **Result:** circular. It only proves the detector agrees with itself.
    - **Gap:** Methodology bug.
    - **Change:** group only by ground-truth labels (`label-welders.py`).

11. **Sequential sweep over about 50 cameras catches welds**
    - **Result:** short welds start and stop between visits. The Orin also ran hot.
    - **Gap:** Medium.
    - **Change:** fewer cameras, and a burst capture at trigger time. Reduce heat with `INFER_MIN_INTERVAL_SEC`, sweep interval and `ANALYZE_WIDTH`, never by cutting annotation.

12. **1 frame every 5 s for 3 min is enough data**
    - **Result:** misses short-lived sparks.
    - **Gap:** Medium.
    - **Change:** `burst/` of every frame for 10 s at event start, then the interval frames.

13. **Weld timer tracks a welder over time**
    - **Result:** tracks poorly.
    - **Gap:** Large.
    - **Change:** open. Needs a tolerance for breaks (about 5 frames) and better tracking (ByteTrack).

14. **Direct RTSP recording works for every camera**
    - **Result:** many cameras refuse.
    - **Gap:** Medium.
    - **Change:** FFmpeg stream copy, falling back to the NVR channel (`record-camera.py`).

15. **Raising thresholds fixes false positives**
    - **Result:** not verified. It trades directly against recall.
    - **Gap:** Open.
    - **Change:** calibrate from labelled data instead of guessing.

## Lessons

- **Try mainstream frames before touching thresholds again.** Most failures (small arcs, far people, smeared sparks) are resolution problems.
- Every "precision" filter (blob floor, saturation, morph open, hard gate) has cost recall. Add one only with labelled evidence that it helps.
- Don't draw conclusions from still frames about checks that need motion.
- Evaluate against labels, never against the detector's own score.
- Colour cues pick up clothing and PPE on this site (blue shirts, yellow hats).

## Next to try

- Run all methods on mainstream frames and compare with substream.
- Hand-drawn ROI for one camera, and test purely on that.
- Per-camera ignore masks for known false-positive spots.
- Shape features: reflections are flat and streak-like, arcs are blobs.
- A small classifier on the brightness/area/contrast features instead of the hand-weighted sum (0.5·contrast + 0.3·distance + 0.2·area).
- A pose model for crouched welders.
