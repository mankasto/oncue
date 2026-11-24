# Parameters

The values the paper states, where each is set, and where it comes from. They are the defaults.
A `Config` field (`Class.field` below, in `src/oncue/core/config.py`) can be overridden with
`oncue --config FILE`; a module constant (an UPPER_CASE name) is fixed in the code, and the
`.settings.json` of every output records the ones the pipeline depends on
(`generation.settings.constants`), so a run resumes only under the same values.
Paths are under `src/oncue/` unless they start with `experiments/` or `tools/`. A source left
blank is a design choice with no recorded measurement behind it.

Log statistics come from `oncue stats` (`data/stats.py`) on the whole WOMD v1.3.1 training set:
1,000 shards, 484,710 scenes, 319,242 moving pedestrians, 44,912 cyclists and 2,683,889 vehicles
(`experiments/stats/womd_stats_train_v1.3.1.json`, with scene-bootstrap intervals). Box sizes are
the medians of 54 v1.2 training shards, equal to the whole set's to 1 cm.

## Visibility and exposure (paper 3.1, supplement A.2)

| Parameter | Value | Set in | Source |
| --- | --- | --- | --- |
| Visible frames that confirm an exposure | 3 | `core/config.py` `VisibilityConfig.min_visible_frames` | |
| Clear rays (of five) | 20 % | `VisibilityConfig.threshold` | |
| Perception range | 80 m | `core/config.py` `PERCEPTION_RANGE_M` | WOMD tracks end at about 80 m in the development set |
| Sensor height | 2.3 m | `VisibilityConfig.sensor_height_m` | the logged ego's measured height, 2.33 m |
| Ray target below the target's top | 0.8 m | `VisibilityConfig.visible_margin_m` | calibrated, below |
| Lowest ray target | 0.3 m | `core/visibility.py` `MIN_AIM_HEIGHT_M` | |
| Static-occluder setback | 4 m | `VisibilityConfig.static_setback_m` | calibrated, below |
| Hidden at least | 0.5 s | `VisibilityConfig.min_occluded_s` | |
| Re-occlusion at most | 0.5 s | `SiteConfig.max_gap_s` | |

Calibration of the margin and the setback: a grid search against WOMD's tracking on 10 training
shards (4,941 scenes, split into a calibration and a check half by drive) picked the pair with
the fewest tracked frames judged hidden plus interior-gap frames judged visible on the
calibration half; the pick held on the check half (scores 0.787 to 0.829, a flat surface). This
measures consistency with WOMD's offline tracking, not visibility accuracy
(`experiments/stats/calibration_train10.json`).

## Requests and starts (paper 3.2, supplement A.2)

The supplement's sensitivity runs cover the protocol choices here: the windows W, the timing
tolerance, the route kinds and angles, how far back a standing target stands, the number of
starts and the start accelerations tried.

| Parameter | Value | Set in | Source |
| --- | --- | --- | --- |
| Starts tried per request | 5 | `generation/targets.py` `TargetSpec.starts_per_request` | protocol choice |
| Assumed history | 1 s | `TargetSpec.assumed_history_s` | |
| On-time tolerance | ±0.2 s | `TargetSpec.timing_tolerance_s` | protocol choice |
| Crossing angles, pedestrians and cyclists | 90°, 60° | `generation/targets.py` `STRAIGHT_ANGLES_DEG` | 60° is 30° off square: the p75 of how far logged pedestrian crossings deviate from perpendicular (31°, median over the lanes each crosses, 1,224 crossings in 4 shards; 28° at the first lane entry over the training set). Not 30° towards the ego, which would run through the hiding car |
| Cyclist pull-out angles | 15°, 30° | `CYCLIST.oblique_angles_deg` | protocol choice, sensitivity on the development set |
| Corner contact overlap | 0.2 m | `generation/targets.py` `CORNER_OVERLAP_M` | protocol choice for pedestrians and cyclists: short windows at common speeds that the NCAP points reach only at a sprint |
| Contact points | NCAP CPNCO, CBNAO, CCCscp | `PEDESTRIAN`, `CYCLIST`, `VEHICLE` in `generation/targets.py` | Euro NCAP AEB VRU protocol v4.5.1 and AEB C2C v4.3 |
| Near miss | post-encroachment time ≤ 1 s above 2 m/s | `generation/targets.py` `UNIFIED_NEAR_MISS` | traffic conflict technique (Allen et al. 1978; Svensson 1998) |
| Stand-back from the hiding site | 2 m, 1 m in a queue gap | `PEDESTRIAN.stand_back_m`, `CYCLIST.stand_back_m`; `generation/motion/starts.py` `QUEUE_BACK_M` | protocol choice |
| Start-position spacing | 0.5 m | `generation/motion/starts.py` `STAND_PLACE_STEP_M` | |
| Grid starts | every 5 m back to 8 s at median speed; standing starts every 0.5 s | `GRID_PLACE_STEP_M`, `GRID_HORIZON_S`, `GRID_GO_STEP_S` | |
| Starting speeds | p25, p50, p75 | `LogMotion.start_speeds_mps` | log statistics: crossing speed in the 1 s before lane entry (pedestrians, cyclists); speed-to-limit ratio (vehicles) |
| Common speeds | p5 to p95 | `LogMotion.speed_common_mps` | log statistics |
| Rare motion | p99 of acceleration, jerk, lateral acceleration and yaw rate | `PEDESTRIAN_KINEMATICS`, `CYCLIST_KINEMATICS`, `VEHICLE_KINEMATICS` | log statistics |
| Peak speed | p99.5; vehicles 24.844 m/s | the same, `VEHICLE_PEAK_SPEED_MPS` | log statistics |
| Too slow | p1 of peak speed | `PEDESTRIAN_MIN_PEAK_SPEED_MPS`, `CYCLIST_MIN_PEAK_SPEED_MPS` | log statistics |
| Peak over required speed | p99 of peak speed over moving speed | `PEDESTRIAN_PACE_RATIO`, `CYCLIST_PACE_RATIO`, `VEHICLE_PACE_RATIO` | log statistics |
| Speeding tolerance | 0.655 m/s | `core/config.py` `SPEEDING_TOL_MPS` (the vehicle spec's and `OracleConfig.speeding_tol_mps`) | p95 of the speed over the posted limit, 2,631,307 moving training vehicles |
| Pedestrian limits | 12.3 m/s, 10 m/s² | `core/config.py` `PEDESTRIAN_LIMITS` | Bolt's peak sprint speed (Stuhec et al. 2023, Table 3); sprint start force plus 2 SD (Stavridis) |
| Cyclist braking | 8.5 m/s² | `core/config.py` `CYCLIST_LIMITS` | bicycle pitch-over at 0.87 g (Mordfin, NBS 1975) |

## Trajectory generator (paper 3.2, supplement A.3)

| Parameter | Value | Set in |
| --- | --- | --- |
| Candidates K | 64 | `models/config.py` `ModelConfig.intention_points` |
| Refinement stages D | 6 | `ModelConfig.decoder_layers` |
| Encoder and decoder | 6 layers, 8 heads, widths 256 and 512, 16 local neighbours | `ModelConfig` |
| Inputs | 11 history frames, 128 neighbours, 768 map chunks of 20 points | `models/datasets/waymo/waymo_dataset.py` `HISTORY_FRAMES`, `NEIGHBOURS`, `MAP_CHUNKS`, `SEGMENT_POINTS` |
| Output | 16 spline controls, 80 positions at 10 Hz | `models/utils/motion_utils.py` `NCTRL`; `ModelConfig.future` |
| Waypoints given in training | 0, 1, 2, 3 with probabilities 0.1, 0.2, 0.5, 0.2 | `TrainConfig.waypoint_counts` |
| Waypoint noise | 0.3 m | `TrainConfig.noise_m` |
| Route and signal dropout | 0.5, 0.2 | `RouteTrainConfig.p_route_off`, `RouteTrainConfig.p_sig_off` |
| Waypoint loss weights (contact, exposure, extra) | 1, 3, 1 | `TrainConfig.slot_weights` |
| Optimizer | AdamW, weight decay 0.01 | `models/train.py`; `TrainConfig.weight_decay` |
| Learning rate | 1e-4, then 2e-5 when fine-tuning (`--lr`) | `TrainConfig.lr`; `cli/model.py` |
| Batch per process, GPUs | 128, 8 | `TrainConfig.batch`; `tools/train.sh` (one process per GPU) |
| Precision, seed | bfloat16 (`--bf16`), 0 | `RouteTrainConfig.bf16`, `RouteTrainConfig.seed` |

## Labels: reference ego, responses and vehicle (supplement B)

| Parameter | Value | Set in | Source |
| --- | --- | --- | --- |
| Route band | ego half width plus 0.4 m per side | `DriverConfig.corridor_margin_m` | |
| Log after the reference contact | at least 1 s | `OracleConfig.cover_contact_s` | |
| Response delay | 0 s | `OracleConfig.reaction_delay_s` | 0.2 s was only a development comparison |
| Reference ego's crossing gap | 1.5 s | `generation/reference.py` `YIELD_GAP_S` | |
| Reference ego's IDM | acceleration 2.5, comfortable and maximum deceleration 3 and 8.3 m/s²; jerk 30 m/s³; headway 1.2 s; standstill gap 2 m; lateral 2 m/s²; stop-sign wait 1 s | `DriverConfig.a_max`, `b_comf`, `b_max`, `jerk_max`, `t_head_s`, `s0_m`, `lateral_accel`, `stop_sign_wait_s` | the jerk is the oracle's emergency brake onset, so no ego brakes harder than the responses assume |
| Response library | 288 core, 1,316 in all | `OracleConfig.budget`; `oracle/responses.py` `candidate_list` | |
| Delayed actions | 0.5 s | `CandidateConfig.delay_s` | |
| Lane width without a mapped one | 3.6 m | `CandidateConfig.default_lane_width_m` | |
| Stop before the conflict | 1 m | `Longitudinal.stop_margin_m` | |
| Wheelbase, centre to rear axle | 2.7 m, 1.35 m | `VehicleConfig.wheelbase`, `VehicleConfig.center_to_rear_axle` | |
| Vehicle limits | acceleration 6, braking 8.3, lateral 6.4 m/s²; curvature 0.3 1/m; brake and lateral jerk 30 and 16 m/s³ | `core/config.py` `VEHICLE_LIMITS` | NIEON (Scanlon et al. 2022, p. 10): braking 8.3 at 30, lateral 6.4 at 16; Waymax (Gulino et al. 2023): acceleration 6, curvature 0.3 |
| Execution per 0.1 s frame | 10 substeps, 5 collision checks | `VehicleConfig.substeps`, `OracleConfig.collision_substeps` | |
| Speed and tracking error | at most 40 m/s and 0.5 m | `OracleConfig.max_speed`, `OracleConfig.max_tracking_error_m` | tracking: NAVSIM's lane-keeping tolerance |

## Evaluation and baselines (supplement C.1, C.2)

| Parameter | Value | Set in | Source |
| --- | --- | --- | --- |
| Intervals | 10,000 bootstrap resamples by driving episode, seed 0, 95 % percentile | `evaluate/bootstrap.py` `RESAMPLES`, `SEED`, `LEVEL`, `METHOD` | |
| Zero-event intervals | exact Clopper-Pearson over the units with a denominator | `evaluate/bootstrap.py` `EXACT_METHOD` | a bootstrap of all-zero draws is always zero |
| Driving episodes | logs whose ego paths share at least 10 frames within 0.05 m, both ways | `DRIVE_SHARED_FRAMES`, `DRIVE_OVERLAP_M` | WOMD stores overlapping windows of one drive as separate logs, tracing one path to about 1e-5 m |
| Closed-loop list | up to 10 tests per site, class, label and W | `experiments/run_all.sh` `PER_STRATUM`; `experiments/e6_closed_loop/testlist.py` `STRATA` | |
| Adversarial reward | 10 (1 - t/N), N = 80; exp(-0.2 d) within 20 m | `baselines/sage_metrics.py` | SAGE's metric code |
| Penalties | background contact x10, road edge x50 | `baselines/realism.py` `CRASH_OBJECT_SCALE`, `CROSS_LINE_SCALE` | SAGE's metric code |
| GOOSE | 5 policy rounds, seed 0 | `baselines/goose/run.py` `POLICY_STEPS`, `SEED` | seed 0 reproduces every GOOSE column of SEAL's published metrics |
| KING | 151 Adam iterations, learning rate 0.005, 4 Hz, 8 s, unit collision and road weights | `baselines/king/run.py` `KING_ARGS`, `HORIZON_S` | |

## Planners and outcomes (supplement C.3)

| Parameter | Value | Set in | Source |
| --- | --- | --- | --- |
| IDM prediction | 3 s, constant velocity | `DriverConfig.predict_s` | |
| Conservative IDM | desired-speed factor 0.30 | `closedloop/registry.py` `CONSERVATIVE_ALPHA` | fit on 93 development scenes: at 0.30 its median no-target 4 s distance matches the occlusion-aware IDM's (they meet at 0.298, 15.0 m) |
| Occlusion-aware IDM | cautious braking 5 m/s² | `closedloop/registry.py` `DRIVER_OPTIONS` | |
| PDM-Closed | proposals at upstream's 3 m/s²; emergency brake at 8.3 m/s² at any speed; offset paths ease in over 2 s | `closedloop/registry.py` `main_pdm_config`; `planners/pdm/planner.py` `OFFSET_RAMP_S` | a 2 s quintic ramp peaks at 1.4 m/s² for a 1 m offset and 2.9 m/s² for 2 m, under the scorer's comfort bound 4.89 and the vehicle limit 6.4 |
| LatentDriver | J = 3 checkpoint, 80 x 20 m crop | `planners/latentdriver/model.py` `NUM_OF_DECODER`; `planners/latentdriver/observation.py` `FOV_M` | |
| Road checks | off road 0.34 m, wrong way 0.412 m, reverse heading 159.6° | `core/road.py` `OFFROAD_TOL_M`, `WRONG_WAY_TOL_M`, `OPPOSING_DEG` | p99.9 of the logged ego on 48,826 scenes from 100 training shards (`experiments/stats/road_train100.json`). Read on the logged ego, not other vehicles, whose tails are flipped box headings; about 2.5 % of logged in-lane cyclists ride against the lane, so cyclists share the rule |
| Brake-onset sensitivity | 30 m/s³ (main), 13.9 m/s³ | `experiments/stages.py` `BRAKE_ONSETS` | NIEON; UN R157's 0.6 s to 0.85 g |
