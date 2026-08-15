# Piper Pick-and-Place with Isaac Sim 5.1.0 and cuRobo

Collision-aware Piper pick-and-place simulation in **NVIDIA Isaac Sim 5.1.0** using **cuRobo**.

The workspace contains the Piper robot model, aligned testbed/table asset, basket asset, cuRobo robot configuration, randomized pick-and-place task logic, collision-aware return-to-home planning, retry handling, and CSV logging.

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
│   └── sim/
│       ├── helper.py
│       └── piper_pick_and_place_testbed_portable.py
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
