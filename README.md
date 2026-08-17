# Piper Pick-and-Place with Isaac Sim 5.1.0 and cuRobo

Collision-aware Piper pick-and-place simulation in **NVIDIA Isaac Sim 5.1.0** using **cuRobo**.

The workspace contains the Piper robot model, aligned testbed/table asset, basket asset, cuRobo robot configuration, randomized pick-and-place task logic, collision-aware return-to-home planning, retry handling, CSV logging, RealSense/ChArUco camera calibration utilities, and optional per-attempt scene-light randomization.

The main simulation entry point resolves repository-local paths from the script location, so the repository directory can be renamed or cloned to another location without editing machine-specific paths in the Python entry point.

---

## Demo

![Piper pick-and-place simulation](docs/media/piper_pick_place_demo.gif)

---

## Features

- Piper robot simulation in Isaac Sim
- Referenced Piper + table testbed USD
- cuRobo collision-aware Cartesian motion planning
- cuRobo joint-space planning for `MOVE_HOME`
- Invisible tabletop collision proxy for planning/physics
- Basket collision handling
- Randomized object spawn pose
- Top-grasp orientation handling
- Joint 6 grasp/transport constraints
- Grasp verification
- Automatic failure handling and retry logic
- Repeated episode execution
- Joint-state CSV logging
- Task-level success/failure logging
- Main and wrist camera support
- RealSense RGB camera extrinsic calibration with a ChArUco board
- Calibrated main-camera pose/intrinsics loading from `configs/camera/main_camera.yaml`
- Per-attempt randomization of four workspace SphereLight intensities
- Optional suppression of the default ground-plane SphereLight contribution
- Repository-relative runtime paths

---

## Tested Environment

This project was tested with the following environment.

| Component | Tested configuration |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-136-generic |
| GPU | NVIDIA GeForce RTX 5080 |
| GPU memory | 16303 MiB |
| NVIDIA Driver | 580.173.02 |
| Isaac Sim | **5.1.0** |
| Python | 3.11.13 |
| PyTorch | 2.7.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuRobo Python package | `nvidia-curobo 0.7.7.post1.dev5` |
| cuRobo Git describe | `v0.7.7-5-gebb7170` |
| cuRobo branch | `main` |
| cuRobo commit | `ebb71702f3f70e767f40fd8e050674af0288abe8` |

A standalone CUDA toolkit (`nvcc`) was not installed in the tested system. The simulation used the CUDA runtime provided through the Isaac Sim / PyTorch environment.

---

## Repository Structure

Important runtime files and directories:

```text
<repo-root>/
├── assets/
│   ├── basket/
│   │   └── basket.usd
│   ├── piper_testbed/
│   │   └── piper_testbed.usd
│   ├── robot/
│   │   └── piper_description/
│   │       ├── meshes/
│   │       └── urdf/
│   │           └── piper_description.urdf
│   └── table/
│       └── table_asset.usd
├── configs/
│   ├── camera/
│   │   └── main_camera.yaml
│   └── curobo_piper/
│       ├── collision_spheres.yml
│       ├── piper.yml
│       └── piper_usd/
│           └── piper_v2.usd
├── docs/
│   └── media/
│       ├── piper_pick_place_demo.gif
│       └── piper_pick_place_demo.mp4
├── logs/
├── src/
│   ├── camera/
│   │   ├── calibration/
│   │   │   └── realsense_charuco_camera_calibration.py
│   │   └── verify_main_camera_from_yaml.py
│   └── sim/
│       ├── helper.py
│       ├── piper_pick_and_place_testbed_portable.py
│       ├── piper_pick_and_place_testbed_calibrated_main_camera.py
│       └── piper_pick_and_place_testbed_calibrated_main_camera_random_lights.py
└── README.md
```

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/botjebi/curobo_piper_v2
cd curobo_piper_v2
```

Run a single-episode smoke test:

```bash
cd src/sim

omni_python piper_pick_and_place_testbed_portable.py \
  --startup_arm_effort 1000 \
  --startup_settle_steps 60 \
  --episode_reset_settle_steps 25 \
  --object_xyz 0.4 0.2 0.025 \
  --place_xyz 0.3 -0.2 0.025 \
  --spawn_region_center_xy 0.25 0.25 \
  --max_episodes 1
```

If this succeeds, remove `--max_episodes 1` or set:

```bash
--max_episodes 0
```

for continuous execution.

---

## Typical Experiment Command

```bash
cd src/sim

omni_python piper_pick_and_place_testbed_portable.py \
  --startup_arm_effort 1000 \
  --startup_settle_steps 60 \
  --episode_reset_settle_steps 25 \
  --object_xyz 0.4 0.2 0.025 \
  --place_xyz 0.3 -0.2 0.025 \
  --spawn_region_center_xy 0.25 0.25
```

---

## Camera Calibration

The repository includes a RealSense RGB camera calibration utility based on a ChArUco board. The calibration estimates the camera pose in the Piper/Isaac world frame and saves the result, together with the active RGB intrinsics, to:

```text
configs/camera/main_camera.yaml
```

The calibrated simulation scripts use this YAML to place `/World/main_camera` and configure its pinhole intrinsics.

Run the calibration utility from the repository root:

```bash
omni_python \
  src/camera/calibration/realsense_charuco_camera_calibration.py
```

After saving a calibration, verify the resulting camera placement in Isaac Sim:

```bash
omni_python \
  src/camera/verify_main_camera_from_yaml.py
```

To run pick-and-place using the calibrated main camera:

```bash
omni_python \
  src/sim/piper_pick_and_place_testbed_calibrated_main_camera.py \
  --startup_arm_effort 1000 \
  --startup_settle_steps 60 \
  --episode_reset_settle_steps 25 \
  --object_xyz 0.4 0.2 0.025 \
  --place_xyz 0.3 -0.2 0.025 \
  --spawn_region_center_xy 0.25 0.25
```

The camera configuration can also be overridden explicitly:

```bash
--main_camera_config_path configs/camera/main_camera.yaml
```

---

## Randomized Scene Lighting

`piper_pick_and_place_testbed_calibrated_main_camera_random_lights.py` extends the calibrated-camera simulation with lighting domain randomization.

Four workspace `SphereLight` sources remain at fixed positions, while their intensities are sampled independently at the beginning of each pick-and-place attempt. The sampled values remain fixed during that attempt and are resampled for the next attempt, including retries.

The default randomized intensity range is:

```text
50000 - 100000
```

The default SphereLight created under `/World/defaultGroundPlane` is kept in the stage but its intensity is set to `0`, so it does not dominate the randomized workspace lighting.

Example:

```bash
omni_python \
  src/sim/piper_pick_and_place_testbed_calibrated_main_camera_random_lights.py \
  --startup_arm_effort 1000 \
  --startup_settle_steps 60 \
  --episode_reset_settle_steps 25 \
  --object_xyz 0.4 0.2 0.025 \
  --place_xyz 0.3 -0.2 0.025 \
  --spawn_region_center_xy 0.25 0.25 \
  --scene_light_intensity_min 50000 \
  --scene_light_intensity_max 100000
```

Useful lighting options:

```text
--scene_light_intensity_min <value>   Minimum randomized intensity
--scene_light_intensity_max <value>   Maximum randomized intensity
--scene_light_random_seed <seed>      Independent reproducible lighting seed
--disable_scene_light_randomization   Use one fixed intensity for all four lights
--scene_light_intensity <value>       Fixed intensity used when randomization is disabled
--disable_scene_lights                Do not create the four workspace lights
--keep_default_light                  Leave the default ground-plane light unchanged
```

During execution, the sampled values are printed for each attempt, for example:

```text
[LIGHT] attempt #3 (success repeat) | mode=random | L1=74231.5, L2=91304.2, L3=52688.7, L4=84112.9
```

---

## Task State Sequence

The main pick-and-place state sequence is:

```text
SNAPSHOT_OBJECT
    ↓
MOVE_PRE_PICK
    ↓
MOVE_PICK
    ↓
CLOSE_AND_LIFT
    ↓
VERIFY_GRASP
    ↓
REORIENT_FOR_TRANSPORT
    ↓
MOVE_PRE_PLACE
    ↓
MOVE_PLACE
    ↓
OPEN_GRIPPER
    ↓
RETREAT
    ↓
MOVE_HOME
    ↓
DONE
```

---

## Logging

Joint-state logs are stored under:

```text
logs/joint_state/
```

The simulation records measured joint positions and velocities.

Task-level logic also retains successful task logs and discards failed task logs according to the configured retry flow.

---

## License

No license is specified yet.

Add a `LICENSE` file before public distribution and update this section accordingly.

---

## Citation

If this repository becomes part of a paper, thesis, or technical report, add the corresponding citation here.
