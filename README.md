# Piper Pick-and-Place with Isaac Sim 5.1.0 and cuRobo

Collision-aware Piper pick-and-place simulation in **NVIDIA Isaac Sim 5.1.0** using **cuRobo**.

The workspace contains the Piper robot model, aligned testbed/table asset, basket asset, cuRobo robot configuration, randomized pick-and-place task logic, collision-aware return-to-home planning, retry handling, and CSV logging.

The main simulation entry point resolves repository-local paths from the script location, so the repository directory can be renamed or cloned to another location without editing machine-specific paths in the Python entry point.

---

## Demo

![Piper pick-and-place simulation](docs/media/piper_pick_place_demo.gif)

[Watch the full simulation video](docs/media/piper_pick_place_demo.mp4)

Recommended media layout:

```text
docs/
└── media/
    ├── piper_pick_place_demo.gif
    └── piper_pick_place_demo.mp4
```

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

## Requirements

This repository does **not** include Isaac Sim or cuRobo.

Install the following separately:

1. NVIDIA GPU driver
2. NVIDIA Isaac Sim 5.1.0
3. cuRobo matching the tested revision
4. Isaac Sim Python launcher

Recommended cuRobo revision:

```text
ebb71702f3f70e767f40fd8e050674af0288abe8
```

If cuRobo was cloned from Git:

```bash
cd <CUROBO_ROOT>
git checkout ebb71702f3f70e767f40fd8e050674af0288abe8
```

---

## Isaac Sim Python Launcher

The tested system used an `omni_python` shell alias pointing to Isaac Sim's Python launcher.

Example:

```bash
alias omni_python='<ISAAC_SIM_ROOT>/python.sh'
```

Depending on the Isaac Sim installation layout, the launcher path may differ.

Verify it with:

```bash
omni_python -c "import sys; print(sys.executable)"
```

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

`helper.py` is required by the main simulation script and should remain in `src/sim/`.

---

## Quick Start

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
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

Repository-local asset/config paths are already configured as defaults, so normally you do **not** need to pass:

```text
--testbed_usd_path
--external_robot_configs_path
--external_asset_path
--robot_urdf_path
--robot_usd_path
--robot_asset_root_path
--basket_usd_path
```

---

## Useful Runtime Options

Object and task:

```text
--object_xyz X Y Z
--object_yaw_deg DEG
--place_xyz X Y Z
--spawn_region_center_xy X Y
--spawn_region_size_xy SX SY
--spawn_z Z
--max_episodes N
```

Robot startup:

```text
--startup_arm_effort VALUE
--startup_settle_steps N
--episode_reset_settle_steps N
```

Planning:

```text
--transport_time_dilation VALUE
--home_time_dilation VALUE
```

Tabletop collision proxy:

```text
--tabletop_proxy_size_xyz SX SY SZ
--tabletop_proxy_local_xyz X Y Z
--disable_tabletop_collision_proxy
--show_tabletop_collision_proxy
```

Logging:

```text
--disable_joint_logging
--joint_log_dir PATH
--joint_log_every_n_steps N
```

See all options:

```bash
omni_python piper_pick_and_place_testbed_portable.py --help
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

### Collision-aware return home

`MOVE_HOME` uses cuRobo joint-space planning through `plan_single_js()` rather than direct per-joint stepping.

This allows the return trajectory to account for obstacles such as the basket.

---

## Path Portability

The main simulation script computes the repository root from its own file location:

```python
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[1]
```

Project-local paths are then derived from `PROJECT_ROOT`, including:

- Piper meshes
- Piper URDF
- Piper cuRobo USD
- testbed USD
- basket USD
- cuRobo config directory
- log directory

The cuRobo robot configuration is also updated at runtime with repository-local values for:

```text
kinematics.usd_path
kinematics.urdf_path
kinematics.asset_root_path
kinematics.external_asset_path
kinematics.external_robot_configs_path
```

Because of this, the repository directory itself may be renamed without changing the Python entry point.

---

## Recommended Portability Test

Before publishing a release, copy the repository to another path and run one episode:

```bash
cp -a <repo-root> /tmp/piper_portability_test

cd /tmp/piper_portability_test/src/sim

omni_python piper_pick_and_place_testbed_portable.py \
  --max_episodes 1
```

At startup, the script prints resolved paths similar to:

```text
[PATH] PROJECT_ROOT=/tmp/piper_portability_test
[PATH] robot_config_dir=/tmp/piper_portability_test/configs/curobo_piper
[PATH] robot_urdf=/tmp/piper_portability_test/assets/robot/...
[PATH] robot_usd=/tmp/piper_portability_test/configs/curobo_piper/...
[PATH] robot_meshes=/tmp/piper_portability_test/assets/robot/...
[PATH] testbed_usd=/tmp/piper_portability_test/assets/piper_testbed/...
[PATH] basket_usd=/tmp/piper_portability_test/assets/basket/...
```

If the copied workspace runs successfully, Python-side path portability is verified.

Also confirm that USD references remain valid after relocation.

---

## Logging

Joint-state logs are stored under:

```text
logs/joint_state/
```

The simulation records measured joint positions and velocities.

Task-level logic also retains successful task logs and discards failed task logs according to the configured retry flow.

For GitHub, generated logs should normally be excluded:

```gitignore
logs/
__pycache__/
*.pyc
```

---

## Environment Verification

Check system information:

```bash
cat /etc/os-release
uname -r
nvidia-smi
```

Check the Isaac Sim Python environment:

```bash
omni_python - <<'PY'
import sys
import torch
import curobo

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuRobo module:", curobo.__file__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Check the cuRobo revision:

```bash
git -C <CUROBO_ROOT> rev-parse HEAD
git -C <CUROBO_ROOT> describe --tags --always --dirty
```

Expected tested commit:

```text
ebb71702f3f70e767f40fd8e050674af0288abe8
```

---

## Troubleshooting

### `omni_python: command not found`

Use the Isaac Sim Python launcher directly or define an alias:

```bash
alias omni_python='<ISAAC_SIM_ROOT>/python.sh'
```

### Robot configuration cannot be loaded

Check:

```text
configs/curobo_piper/piper.yml
```

### Piper URDF or meshes cannot be found

Check:

```text
assets/robot/piper_description/urdf/piper_description.urdf
assets/robot/piper_description/meshes/
```

### Testbed does not load

Check:

```text
assets/piper_testbed/piper_testbed.usd
```

and verify that its referenced USD assets are still resolvable after cloning or moving the repository.

### Basket does not load

Check:

```text
assets/basket/basket.usd
```

### CUDA / GPU errors

Compare against the tested environment:

```text
Isaac Sim: 5.1.0
NVIDIA Driver: 580.173.02
PyTorch: 2.7.0+cu128
PyTorch CUDA runtime: 12.8
cuRobo commit: ebb71702f3f70e767f40fd8e050674af0288abe8
```

A separate system `nvcc` installation was not required in the tested environment.

---

## GitHub Media

The recommended README preview is the GIF:

```markdown
![Piper pick-and-place simulation](docs/media/piper_pick_place_demo.gif)
```

The full MP4 is linked separately:

```markdown
[Watch the full simulation video](docs/media/piper_pick_place_demo.mp4)
```

Place both files under:

```text
docs/media/
```

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]

# Logs
logs/

# Editor / OS
.vscode/
.idea/
.DS_Store

# Temporary files
*.tmp
*.bak
*~
```

If generated Isaac Sim cache folders or large temporary outputs are added later, exclude them as well.

---

## Notes for Public Release

Before publishing the repository:

1. Run the portability smoke test from a different directory.
2. Check USD references for machine-specific absolute paths.
3. Remove or relocate legacy conversion/debug scripts that contain local development paths.
4. Confirm which CAD/source assets may legally be redistributed.
5. Add a project license.
6. Consider Git LFS if large binary CAD/USD/mesh files exceed normal GitHub limits.

---

## License

No license is specified yet.

Add a `LICENSE` file before public distribution and update this section accordingly.

---

## Citation

If this repository becomes part of a paper, thesis, or technical report, add the corresponding citation here.
