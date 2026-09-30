# Football Player Detection & Tracking ⚽

End-to-end football video analysis pipeline built on YOLO and OpenCV. Detects players, goalkeepers, referees, and the ball; assigns teams via jersey color clustering; computes pitch homography for 2D tactical mapping; and tracks possession, passes, and turnovers — all in a single pass over the video.

---

## 🎥 Demos

### YOLO Tracking
Player, goalkeeper, and referee detection with BoT-SORT tracking, ball trail overlay, and possession indicator.

[![YOLO Tracking Demo](https://github.com/Simo-03/football-player-detection/releases/download/v1.0.0/yolo_demo_preview.gif)](https://github.com/Simo-03/football-player-detection/releases/download/v1.0.0/YOLO-demo.mp4)

### Tactical View
Team-colored ground markers, 2D pitch minimap, possession bar, pass/turnover counts, and ball speed — all in a single split-panel output.

[![Tactical View Demo](https://github.com/Simo-03/football-player-detection/releases/download/v1.0.0/tactical_demo_preview.gif)](https://github.com/Simo-03/football-player-detection/releases/download/v1.0.0/tactical-analysis-demo.mp4)

---

## 🤖 Models

| File | Purpose | Architecture | Input Size |
|------|---------|--------------|------------|
| `best_players_gk_1280_s_e300.pt` | Player / GK / Referee | YOLO11s | 1280 |
| `pitch_kpts32_y8s_640_e500_AO.pt` | Pitch keypoints (32 landmarks) | YOLOv8s-pose | 640 (run at 960) |
| `ball_tracking_1280_e300.pt` | Ball detector (sliced inference) | YOLO11s | 640x640 tiles |

Place model weights in the `models/` directory.

---

## 📊 Results

Headline metrics from the final-year dissertation evaluation. Detection mAP values are from held-out Roboflow validation sets; pipeline metrics are from a 30-second DFL Bundesliga broadcast clip processed at full stride on Apple Silicon MPS (750 frames).

| Area | Metric | Result | Notes |
|------|--------|--------|--------|
| Player/GK/referee detection | mAP50 | 0.923 | YOLO11s, held-out validation set |
| Dedicated ball detection | mAP50 | 0.929 | YOLO11s with sliced inference |
| Ball detection coverage | Frames with ball | 633 / 750 (84.4%) | 30-second broadcast test clip |
| Tracking continuity | Player tracks | 23 distinct tracks, median 750 frames | BoT-SORT with sparse optical-flow GMC |
| Homography stability | Valid projection | 693 / 750 frames (92.4%) | 15 additional fallback frames (2.0%) |
| Team assignment | Manual sample accuracy | 100 / 100 player observations | Five annotated frames; one transient error seen outside sample |
| Possession/pass analytics | Clip summary | Team 1: 54% possession, Team 0: 46%; 7 passes, 2 turnovers | Rule-based event estimates |

Not yet reported: end-to-end FPS, full MOTChallenge metrics (IDF1/MOTA/HOTA over a longer labelled sequence), independent ground-truth homography reprojection error, and pass/possession precision-recall. Those should be tracked in future evaluation runs before making stronger real-time or tactical-accuracy claims.

---

## 🧩 Key Engineering Challenges

**Camera movement and homography stability.** Broadcast footage shifts constantly, so a homography solved from one frame can fail when pitch keypoints are sparse, noisy, or clustered in a small region. The pipeline uses `HomographyTransformer` in `src/analytics/homography_simple.py`, which calls OpenCV RANSAC homography estimation, then the main loop gates each candidate by inlier count, inlier ratio, pitch span, and median reprojection error before accepting it. When a candidate fails, the pipeline reuses the previous valid transform for up to 60 frames and clamps projected player-position jumps before smoothing, preventing short keypoint failures from creating tactical-map jumps.

**Tracking robustness through occlusion.** Players overlap frequently, leave the frame, and re-enter near similar-looking teammates, which makes ID continuity difficult even when detections are good. `src/vision/tracker.py` runs Ultralytics tracking with the committed BoT-SORT config in `configs/botsort.yaml`, using persistent tracks, confidence/IoU thresholds, a 30-frame track buffer, score fusion, and sparse optical-flow global motion compensation to keep IDs stable across camera motion and brief occlusions. The default config currently has `with_reid: False`, so appearance ReID is best treated as future work rather than a claimed feature of the committed pipeline.

---

## 💻 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Simo-03/football-player-detection.git
   cd football-player-detection
   ```

2. **Install uv** (Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Download model weights**
   ```bash
   bash scripts/download_models.sh
   ```
   Or manually: go to [Releases](https://github.com/Simo-03/football-player-detection/releases/latest), download the `.pt` files, and place them in `models/`.

5. **Add an input video** to `assets/`

6. **Run the pipeline**
   ```bash
   uv run python main.py --video assets/your_video.mp4 --run-name my_run
   ```

7. **Check outputs** in `outputs/video_tracking/<run-name>/`:
   - `yolo_players_referees_goalkeepers_ball.mp4` — raw detections + ball trail
   - `team_assignment_homography_minimap.mp4` — team colors + analytics panel

---

## 🛠️ Usage

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Path to input video |
| `--run-name` | video stem | Name for the output subdirectory |
| `--device` | `mps` | Device for player tracker model |
| `--ball-device` | `mps` | Device for ball model |
| `--pitch-device` | `cpu` | Device for pitch keypoint model |
| `--vid-stride` | `1` | Process every Nth frame (2 = 2× faster) |
| `--disable-team-assignment` | off | Skip K-Means team clustering |
| `--disable-possession` | off | Skip ball possession tracking |
| `--debug-keypoints` | off | Overlay pitch keypoint debug on tactical video |
| `--debug-homography-diag` | off | Print per-frame homography diagnostics |
| `--homography-keypoint-conf-threshold` | `0.5` | Keypoint confidence gate |
| `--homography-min-points` | `6` | Min valid keypoints to compute homography |
| `--homography-max-stale-frames` | `60` | Max frames to reuse stale homography |
| `--heatmaps N` | `0` | Generate heatmaps for top N players |
| `--heatmap-track ID` | — | Generate heatmap for specific track ID (repeatable) |
| `--trajectories` | off | Generate trajectory images (all players, per-team, top-k) |
| `--trajectory-track ID` | — | Generate trajectory for specific track ID (repeatable) |
| `--trajectory-top-k` | `3` | Top-k players by distance for trajectory view |
| `--report` | off | Generate player/team summary CSVs |
| `--pass-network` | off | Generate per-team pass network PNGs |

### Example Commands

**Basic run:**
```bash
uv run python main.py --video assets/match.mp4
```

**Fast preview (skip every other frame, MPS acceleration):**
```bash
uv run python main.py --video assets/match.mp4 --vid-stride 2 --device mps --ball-device mps
```

**Skip team assignment:**
```bash
uv run python main.py --video assets/match.mp4 --disable-team-assignment
```

**Relaxed homography (for difficult camera angles):**
```bash
uv run python main.py --video assets/match.mp4 \
  --homography-keypoint-conf-threshold 0.3 \
  --homography-min-points 4
```

**Full post-processing suite:**
```bash
uv run python main.py --video assets/match.mp4 \
  --heatmaps 5 --trajectories --report --pass-network
```

---

## 📁 Project Structure

```
src/
├── app/
│   └── app.py                  # run_video_pipeline() — main frame loop
├── core/
│   ├── types.py                # Detection, Track, FrameData dataclasses
│   └── pitch.py                # SoccerPitchConfiguration (32 vertices, cm)
├── vision/
│   ├── tracker.py              # YOLO11s player/GK/referee tracker
│   ├── pitch_keypoint_detector.py  # YOLOv8s-pose pitch keypoint detector
│   └── ball.py                 # BallTracker + BallAnnotator (sliced inference)
├── analytics/
│   ├── homography_simple.py    # HomographyTransformer (RANSAC, used by pipeline)
│   ├── team_assignment.py      # TeamAssigner (K-Means on LAB color)
│   ├── possession.py           # PossessionTracker + TeamPossessionTracker
│   ├── pass_detection.py       # PassDetector (passes + turnovers)
│   └── player_movement.py      # PlayerMovementAnalyzer (distance, speed)
└── io/
    ├── minimap_renderer.py     # MinimapRenderer + split panel with analytics
    ├── csv_writer.py           # CsvWriter (per-frame track data)
    ├── generate_player_heatmap.py      # Post-processing: player heatmap PNGs
    ├── generate_pitch_trajectories.py  # Post-processing: trajectory PNGs
    ├── generate_match_report.py        # Post-processing: player/team CSVs
    └── generate_pass_network.py        # Post-processing: pass network PNGs

configs/
├── botsort.yaml               # Default tracker config (BoT-SORT)
└── bytetrack.yaml             # Alternate tracker config

models/                        # YOLO model weights (not committed)

scripts/
└── download_models.sh         # Download model weights from GitHub release
```

---

## © License

Licensed under MIT — see the `LICENSE` file. Originally developed as an undergraduate final year project.

## Author

Selim Sherif — LinkedIn: `www.linkedin.com/in/selimsherif`
