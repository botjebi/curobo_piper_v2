try:
    import isaacsim
except ImportError:
    pass

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------
# Project-local paths
# ---------------------------------------------------------------------
# This script is expected at:
#   <PROJECT_ROOT>/src/sim/<script>.py
#
# Resolve assets/configs/logs from the repository location instead of a
# machine-specific path such as /home/lab/icros_journal.
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parents[1]

ASSETS_DIR = PROJECT_ROOT / "assets"
CUROBO_CONFIG_DIR = PROJECT_ROOT / "configs" / "curobo_piper"
LOGS_DIR = PROJECT_ROOT / "logs"

PIPER_MESH_DIR = ASSETS_DIR / "robot" / "piper_description" / "meshes"
PIPER_URDF_PATH = (
    ASSETS_DIR
    / "robot"
    / "piper_description"
    / "urdf"
    / "piper_description.urdf"
)
PIPER_CUROBO_USD_PATH = CUROBO_CONFIG_DIR / "piper_usd" / "piper_v2.usd"
TESTBED_USD_PATH = ASSETS_DIR / "piper_testbed" / "piper_testbed.usd"
BASKET_USD_PATH = ASSETS_DIR / "basket" / "basket.usd"
JOINT_LOG_DIR = LOGS_DIR / "joint_state"

if torch.cuda.is_available():
    _ = torch.zeros(4, device="cuda:0")

parser = argparse.ArgumentParser(
    description=(
        "Physical top-grasp pick-and-place test for Piper in Isaac Sim using cuRobo and a referenced Piper-table testbed USD. "
        "The target object is a horizontal rectangular prism whose long side lies on the table. "
        "The gripper first moves above the object center with joint6 kept near the transport angle, "
        "then aligns joint6 to a canonical grasp yaw that treats object yaw n and n+180 deg as equivalent, "
        "grasping from above the object center before returning joint6 to the transport angle for place."
    )
)
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--robot", type=str, default="piper.yml")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=str(PIPER_MESH_DIR),
    help="cuRobo external asset root used to resolve Piper meshes.",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=str(CUROBO_CONFIG_DIR),
    help="Directory containing piper.yml and related cuRobo robot configs.",
)
parser.add_argument(
    "--robot_urdf_path",
    type=str,
    default=str(PIPER_URDF_PATH),
    help="URDF used by cuRobo kinematics. This does not import a second Isaac Sim robot.",
)
parser.add_argument(
    "--robot_usd_path",
    type=str,
    default=str(PIPER_CUROBO_USD_PATH),
    help="Project-local Piper USD path written into robot_cfg['kinematics']['usd_path'] for cuRobo.",
)
parser.add_argument(
    "--robot_asset_root_path",
    type=str,
    default=str(PIPER_MESH_DIR),
    help="Mesh root written into robot_cfg['kinematics']['asset_root_path'] for cuRobo.",
)
parser.add_argument(
    "--testbed_usd_path",
    type=str,
    default=str(TESTBED_USD_PATH),
    help="Piper + table testbed USD. It is referenced once at --testbed_prim_path.",
)
parser.add_argument(
    "--testbed_prim_path",
    type=str,
    default="/World/robot_testbed",
    help="Stage prim path where the testbed USD is referenced.",
)
parser.add_argument(
    "--tabletop_proxy_size_xyz",
    nargs=3,
    metavar=("sx", "sy", "sz"),
    type=float,
    default=[0.8, 0.74, 0.005],
    help="Simplified tabletop collision proxy size in meters. Defaults to the measured 800 x 740 x 5 mm top plate.",
)
parser.add_argument(
    "--tabletop_proxy_local_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.0, 0.0, -0.0025],
    help="Table-local center of the tabletop proxy. Table origin is the tabletop top-surface center, so z=-2.5 mm for a 5 mm plate.",
)
parser.add_argument(
    "--disable_tabletop_collision_proxy",
    action="store_true",
    default=False,
    help="Do not create the simplified invisible tabletop PhysX/cuRobo proxy. Use only if the table asset already has suitable collision geometry.",
)
parser.add_argument(
    "--show_tabletop_collision_proxy",
    action="store_true",
    default=False,
    help="Render the simplified tabletop collision proxy. By default it remains physically active but invisible.",
)
parser.add_argument(
    "--disable_scene_lights",
    action="store_true",
    default=False,
    help="Do not create the four workspace fill lights.",
)
parser.add_argument(
    "--scene_light_intensity",
    type=float,
    default=3000.0,
    help="Intensity of each of the four workspace SphereLights.",
)
parser.add_argument(
    "--scene_light_radius",
    type=float,
    default=0.25,
    help="Radius in meters of each workspace SphereLight.",
)
parser.add_argument("--reactive", action="store_true", default=False)
parser.add_argument("--visualize_spheres", action="store_true", default=False)
parser.add_argument("--constrain_grasp_approach", action="store_true", default=False)
parser.add_argument(
    "--object_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.42, 0.00, 0.025],
    help="Initial object center position in the world frame. For a horizontal object, z should usually be object_short_size/2.",
)
parser.add_argument(
    "--object_yaw_deg",
    type=float,
    default=0.0,
    help="Initial object yaw around the world z axis in degrees.",
)
parser.add_argument(
    "--place_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.56, -0.18, 0.025],
    help="Target place center position for the object in the world frame.",
)
parser.add_argument(
    "--place_object_yaw_deg",
    type=float,
    default=0.0,
    help="Desired upright object yaw at the place position in degrees.",
)
parser.add_argument(
    "--orientation",
    nargs=4,
    metavar=("w", "x", "y", "z"),
    type=float,
    default=[0.0, 1.0, 0.0, 0.0],
    help=(
        "Base end-effector quaternion [w x y z] for top grasp. "
        "Adjust if your Piper end-effector frame differs."
    ),
)
parser.add_argument("--object_long_size", type=float, default=0.08, help="Long dimension of the horizontal rectangular prism in meters.")
parser.add_argument("--object_short_size", type=float, default=0.05, help="Short dimension of the horizontal rectangular prism in meters. The vertical height is also this value.")
parser.add_argument("--object_mass", type=float, default=0.02)
parser.add_argument(
    "--grasp_height_offset",
    type=float,
    default=0.12,
    help="Offset from the object center to the gripper_base target during top grasp/place.",
)
parser.add_argument(
    "--approach_height",
    type=float,
    default=0.24,
    help="Absolute height offset above the object/place center for pre-pick and pre-place.",
)
parser.add_argument(
    "--lift_delta",
    type=float,
    default=0.05,
    help="Additional vertical lift after grasp before transport reorientation.",
)
parser.add_argument(
    "--transport_joint6_deg",
    type=float,
    default=0.0,
    help="Transport/reference angle for joint6 in degrees. The robot stays close to this value while moving to pre-grasp and while transporting to place.",
)
parser.add_argument(
    "--joint6_transport_limit_deg",
    type=float,
    default=1.0,
    help="Allowed range around the transport joint6 angle during pre-grasp transport and place transport.",
)
parser.add_argument(
    "--joint6_grasp_limit_deg",
    type=float,
    default=1.0,
    help="Allowed range around the canonical grasp joint6 angle during the grasp and lift phases.",
)
parser.add_argument(
    "--joint6_hold_effort",
    type=float,
    default=5000.0,
    help="Maximum effort used by the joint6 limiter while enforcing the active range.",
)
parser.add_argument(
    "--joint6_align_step_deg",
    type=float,
    default=4.0,
    help="Maximum joint6 command step in degrees per simulation iteration while aligning to the grasp/transport angle.",
)
parser.add_argument(
    "--joint6_align_tolerance_deg",
    type=float,
    default=1.0,
    help="Tolerance in degrees used to decide whether joint6 has reached its alignment target.",
)
parser.add_argument(
    "--lock_grasp_branch_during_prepick",
    action="store_true",
    default=True,
    help=(
        "Lock the 180-degree-equivalent grasp yaw branch once at the beginning of MOVE_PRE_PICK, "
        "then keep solving joint6 against that same world-yaw branch during the whole merged pre-pick phase."
    ),
)
parser.add_argument(
    "--disable_lock_grasp_branch_during_prepick",
    action="store_false",
    dest="lock_grasp_branch_during_prepick",
    help="Disable branch locking and revert to per-step branch reselection during MOVE_PRE_PICK.",
)
parser.add_argument(
    "--grasp_direction_mode",
    type=str,
    choices=["top_grasp_short_axis", "top_grasp_long_axis"],
    default="top_grasp_short_axis",
    help=(
        "Object-relative grasp direction mode. "
        "top_grasp_short_axis means the jaws close across the object's short axis from above "
        "(no extra 90 deg offset in the world-yaw target). "
        "top_grasp_long_axis means the jaws close across the object's long axis from above "
        "(implemented as an additional +90 deg offset from the object yaw target)."
    ),
)
parser.add_argument(
    "--joint6_world_yaw_sign",
    type=float,
    default=-1.0,
    help=(
        "Sign used to map the desired jaw-closing yaw in the world frame to the joint6 target. "
        "Use -1 when positive joint6 rotation appears opposite to positive world-z yaw."
    ),
)
parser.add_argument(
    "--joint6_yaw_calibration_offset_deg",
    type=float,
    default=0.0,
    help=(
        "Constant calibration offset added after grasp_direction_mode when converting the desired world yaw "
        "into the joint6 target."
    ),
)
parser.add_argument(
    "--grasp_yaw_trim_deg",
    type=float,
    default=0.0,
    help="Small extra tuning offset in degrees added to the desired jaw-closing yaw before converting it to joint6.",
)
parser.add_argument(
    "--ee_jaw_yaw_offset_deg",
    type=float,
    default=0.0,
    help=(
        "Fixed yaw offset in degrees from the ee_prim world yaw to the actual jaw-closing axis world yaw. "
        "Use this when the gripper jaw direction is not aligned with the ee_prim x/y axis used by yaw_from_quat_wxyz."
    ),
)
parser.add_argument(
    "--gripper_base_above_top",
    type=float,
    default=0.1175,
    help=(
        "Vertical offset from the rectangular prism top face to the gripper_base target during top grasp/place. "
        "This is shape-aware and should usually be used instead of interpreting grasp_height_offset from the object center."
    ),
)
parser.add_argument(
    "--wait_steps_after_gripper",
    type=int,
    default=25,
    help="Simulation steps to wait right after opening/closing the gripper.",
)
parser.add_argument(
    "--post_gripper_settle_steps",
    type=int,
    default=30,
    help="Extra settle steps after open/close before continuing the state machine.",
)
parser.add_argument(
    "--arm_static_threshold",
    type=float,
    default=0.2,
    help="Arm-only velocity threshold for deciding whether a new plan can start.",
)
parser.add_argument(
    "--settle_threshold",
    type=float,
    default=0.05,
    help="Arm-only velocity threshold used while settling after gripper actions.",
)
parser.add_argument(
    "--transport_time_dilation",
    type=float,
    default=0.25,
    help="Slows down MOVE_PRE_PLACE, MOVE_PLACE, and RETREAT planning to reduce slip during transport.",
)
parser.add_argument(
    "--post_place_retreat_height",
    type=float,
    default=0.15,
    help=(
        "Deprecated compatibility option. The testbed version now restores the legacy-sized RETREAT "
        "and uses cuRobo joint-space planning for MOVE_HOME."
    ),
)
parser.add_argument(
    "--home_time_dilation",
    type=float,
    default=0.5,
    help="Time-dilation factor used for cuRobo joint-space planning during MOVE_HOME.",
)
parser.add_argument(
    "--solver_pos_iters",
    type=int,
    default=124,
    help="PhysX solver position iterations applied to the object and robot when available.",
)
parser.add_argument(
    "--solver_vel_iters",
    type=int,
    default=4,
    help="PhysX solver velocity iterations applied to the object and robot when available.",
)

parser.add_argument(
    "--arm_drive_kp",
    type=float,
    default=1047.19751,
    help=(
        "Position-drive stiffness for joint1..joint6. The default matches the legacy helper.py URDF importer."
    ),
)
parser.add_argument(
    "--arm_drive_kd",
    type=float,
    default=52.35988,
    help=(
        "Position-drive damping for joint1..joint6. The default matches the legacy helper.py URDF importer."
    ),
)

parser.add_argument(
    "--startup_arm_effort",
    type=float,
    default=1200.0,
    help="Temporary max effort applied to arm joints during startup/reset settling to reduce initial vibration.",
)
parser.add_argument(
    "--nominal_arm_effort",
    type=float,
    default=5000.0,
    help="Nominal max effort applied to arm joints after the initial settling window.",
)
parser.add_argument(
    "--startup_settle_steps",
    type=int,
    default=50,
    help="How many simulation steps to wait after the very first articulation initialization before starting episodes.",
)
parser.add_argument(
    "--episode_reset_settle_steps",
    type=int,
    default=20,
    help="How many simulation steps to wait after each episode reset before allowing motion planning.",
)
parser.add_argument(
    "--gripper_hold_effort",
    type=float,
    default=2000.0,
    help="Maximum effort used for the gripper finger joints while holding the object.",
)
parser.add_argument(
    "--gripper_hold_kp",
    type=float,
    default=6000.0,
    help="Optional finger position stiffness used while holding, when the API is available.",
)
parser.add_argument(
    "--gripper_hold_kd",
    type=float,
    default=180.0,
    help="Optional finger position damping used while holding, when the API is available.",
)
parser.add_argument(
    "--disable_continuous_gripper_hold",
    action="store_true",
    default=False,
    help="Disable the per-step closed-position hold on the finger joints. Useful only for A/B testing.",
)
parser.add_argument(
    "--grasp_min_lift",
    type=float,
    default=0.015,
    help="Minimum object z increase required after the lift stage to accept the grasp.",
)
parser.add_argument(
    "--grasp_max_ee_error",
    type=float,
    default=0.10,
    help="Maximum allowed distance between the actual object center and the carried-center estimate from the gripper.",
)
parser.add_argument(
    "--release_max_place_error",
    type=float,
    default=0.08,
    help="Maximum allowed XY distance from the place marker after release to report success.",
)
parser.add_argument(
    "--joint_log_dir",
    type=str,
    default=str(JOINT_LOG_DIR),
    help="Directory where numbered joint-state CSV logs are stored.",
)
parser.add_argument(
    "--joint_log_prefix",
    type=str,
    default="piper_rect_top_grasp_joint_state_deg_log",
    help="Filename prefix used for numbered joint-state CSV logs.",
)
parser.add_argument(
    "--joint_log_every_n_steps",
    type=int,
    default=1,
    help="Write one joint-state row every N simulation steps while the simulation is playing.",
)
parser.add_argument(
    "--disable_joint_logging",
    action="store_true",
    default=False,
    help="Disable CSV logging of measured joint states.",
)

parser.add_argument(
    "--spawn_region_center_xy",
    nargs=2,
    metavar=("x", "y"),
    type=float,
    default=None,
    help="Center of the random spawn region in XY. Defaults to object_xyz[:2].",
)
parser.add_argument(
    "--spawn_region_size_xy",
    nargs=2,
    metavar=("sx", "sy"),
    type=float,
    default=[0.12, 0.12],
    help="Full width and full depth of the random spawn region in meters.",
)
parser.add_argument(
    "--spawn_z",
    type=float,
    default=None,
    help="Object center height used during respawn. Defaults to object_xyz[2].",
)
parser.add_argument(
    "--episode_pause_steps",
    type=int,
    default=40,
    help="How many simulation steps to wait after a successful or failed episode before the next episode starts.",
)
parser.add_argument(
    "--snapshot_wait_steps",
    type=int,
    default=40,
    help="How many simulation steps to wait after respawn before the object pose is snapshotted.",
)
parser.add_argument(
    "--spawn_settle_lin_vel_thresh",
    type=float,
    default=0.06,
    help="Linear-velocity threshold used to decide whether a respawned object is settled enough to snapshot.",
)
parser.add_argument(
    "--spawn_settle_ang_vel_thresh",
    type=float,
    default=0.10,
    help="Angular-velocity threshold used to decide whether a respawned object is settled enough to snapshot.",
)
parser.add_argument(
    "--spawn_settle_pos_eps",
    type=float,
    default=0.0015,
    help="Pose-stability threshold on object position change between steps during spawn settling.",
)
parser.add_argument(
    "--spawn_settle_yaw_eps_deg",
    type=float,
    default=1.0,
    help="Pose-stability threshold on object yaw change between steps during spawn settling.",
)
parser.add_argument(
    "--spawn_settle_stable_steps",
    type=int,
    default=8,
    help="How many consecutive stable steps are required before the respawned object is snapshotted.",
)
parser.add_argument(
    "--spawn_settle_timeout_steps",
    type=int,
    default=120,
    help="Maximum number of post-respawn steps to wait before force-accepting the object pose.",
)
parser.add_argument(
    "--spawn_force_zero_steps",
    type=int,
    default=8,
    help="How many early post-respawn steps should explicitly zero the object velocities.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=0,
    help="Maximum number of successful episodes to run. Use 0 for unlimited repetition.",
)
parser.add_argument(
    "--repeat_on_failure",
    action="store_true",
    default=False,
    help="When enabled, a failed episode is also reset and retried with a new random object pose.",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=0,
    help="Seed used for object randomization. Set a different value for a different repeatable sequence.",
)
parser.add_argument(
    "--show_spawn_region",
    action="store_true",
    default=False,
    help="Draw a thin visual guide for the random spawn region when the local Isaac Sim build supports it.",
)


parser.add_argument(
    "--table_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.3, 0.0, -0.35],
    help="Legacy compatibility option; ignored by the testbed-based loader.",
)
parser.add_argument(
    "--table_scale_xyz",
    nargs=3,
    metavar=("sx", "sy", "sz"),
    type=float,
    default=[0.8, 1.6, 0.7],
    help="Legacy compatibility option; ignored by the testbed-based loader. Use --tabletop_proxy_size_xyz instead.",
)
parser.add_argument(
    "--ground_z",
    type=float,
    default=-0.7,
    help="Z position of the default ground plane so the table remains visible above it.",
)
parser.add_argument(
    "--basket_usd_path",
    type=str,
    default=str(BASKET_USD_PATH),
    help="USD asset path for the basket visual/mesh object referenced at /World/basket.",
)
parser.add_argument(
    "--basket_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.3, -0.2, 0.0],
    help="World translate applied to the basket reference prim at /World/basket.",
)
parser.add_argument(
    "--place_marker_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[0.3, -0.2, 0.25],
    help="World translate of the place-marker Xform used as the place goal reference.",
)
parser.add_argument(
    "--place_goal_fixed_z",
    type=float,
    default=None,
    help="Fixed object-center Z used for the place goal when using the place-marker Xform for XY only. Defaults to place_xyz[2].",
)
parser.add_argument(
    "--attached_place_approach_height",
    type=float,
    default=0.04,
    help="Vertical offset above the attached-object release EE goal used for MOVE_PRE_PLACE. Keep this modest because the object is released above the basket, not inserted to the bottom.",
)
parser.add_argument(
    "--basket_release_object_center_z",
    type=float,
    default=0.13,
    help="Absolute world Z for the object's center at the release point above the basket. This avoids pushing the gripper deep into the basket.",
)
parser.add_argument(
    "--use_place_marker_xyz_for_release",
    action="store_true",
    default=False,
    help="Use the full place-marker XYZ as the attached-object release goal. By default only the marker XY is used and Z comes from --basket_release_object_center_z.",
)
parser.add_argument(
    "--disable_curobo_attach_detach",
    action="store_true",
    default=False,
    help="Disable best-effort cuRobo attach/detach of the grasped object during transport.",
)
parser.add_argument(
    "--basket_collision_approximation",
    type=str,
    default="none",
    choices=["none", "convexHull", "convexDecomposition", "meshSimplification", "boundingCube", "boundingSphere"],
    help="Mesh-collision approximation applied to basket meshes when enabling collision. 'none' keeps triangle-mesh collision for a static basket.",
)
parser.add_argument(
    "--home_return_step_deg",
    type=float,
    default=4.0,
    help="Maximum joint step in degrees per simulation iteration while returning the arm to the default retract configuration after each successful task.",
)
parser.add_argument(
    "--home_return_tolerance_deg",
    type=float,
    default=1.0,
    help="Per-joint tolerance in degrees used to decide whether the arm has finished returning to the default retract configuration.",
)

parser.add_argument(
    "--main_camera_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[1.0, 0.0, 0.85],
    help="World translate of the main camera created directly under /World.",
)
parser.add_argument(
    "--main_camera_rpy_deg",
    nargs=3,
    metavar=("roll", "pitch", "yaw"),
    type=float,
    default=[0.0, 40.0, 90.0],
    help="Main camera XYZ Euler rotation in degrees.",
)
parser.add_argument(
    "--wrist_camera_xyz",
    nargs=3,
    metavar=("x", "y", "z"),
    type=float,
    default=[-0.075, 0.0, 0.04],
    help="Local translate of the wrist camera created directly under the testbed Piper gripper_base.",
)
parser.add_argument(
    "--wrist_camera_rpy_deg",
    nargs=3,
    metavar=("roll", "pitch", "yaw"),
    type=float,
    default=[180.0, -40.0, 90.0],
    help="Local XYZ Euler rotation of the wrist camera created directly under the testbed Piper gripper_base.",
)
parser.add_argument(
    "--main_camera_focal_length",
    type=float,
    default=24.0,
    help="Focal length in mm for the main camera prim.",
)
parser.add_argument(
    "--wrist_camera_focal_length",
    type=float,
    default=18.0,
    help="Focal length in mm for the wrist camera prim.",
)

args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {"headless": args.headless_mode is not None, "width": "1920", "height": "1080"}
)

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from helper import add_extensions
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

_SEARCH_DIRS = [_THIS_DIR, _THIS_DIR.parent, PROJECT_ROOT]
for _d in [p.resolve() for p in _SEARCH_DIRS]:
    if str(_d) not in sys.path:
        sys.path.append(str(_d))


class JointStateLogger:
    def __init__(self, csv_path, dof_names, log_every_n_steps=1):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.dof_names = list(dof_names)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.file = self.csv_path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.rows_written = 0
        self.writer.writerow(
            ["sim_step", "sim_time_sec", "state"]
            + [f"{name}_deg" for name in self.dof_names]
            + [f"{name}_vel_deg_s" for name in self.dof_names]
        )
        self.file.flush()

    def maybe_log(self, step_index, sim_time, state, sim_js):
        if step_index % self.log_every_n_steps != 0:
            return
        pos_deg = np.rad2deg(np.array(sim_js.positions, dtype=np.float64))
        vel_deg = np.rad2deg(np.array(sim_js.velocities, dtype=np.float64))
        self.writer.writerow(
            [int(step_index), float(sim_time), str(state)]
            + pos_deg.tolist()
            + vel_deg.tolist()
        )
        self.rows_written += 1
        if self.rows_written % 50 == 0:
            self.file.flush()

    def close(self):
        try:
            self.file.flush()
        except Exception:
            pass
        try:
            self.file.close()
        except Exception:
            pass


def make_numbered_log_path(log_dir, prefix):
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.csv$")
    max_index = 0
    for p in log_dir.glob(f"{prefix}_*.csv"):
        match = pattern.match(p.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    next_index = max_index + 1
    return log_dir / f"{prefix}_{next_index}.csv", next_index


def make_numbered_directory(root_dir, prefix):
    root_dir = Path(root_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_index = 0
    for p in root_dir.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                max_index = max(max_index, int(m.group(1)))
    next_index = max_index + 1
    sim_dir = root_dir / f"{prefix}_{next_index}"
    sim_dir.mkdir(parents=True, exist_ok=False)
    return sim_dir, next_index


def make_task_log_path(sim_dir):
    sim_dir = Path(sim_dir).expanduser().resolve()
    sim_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^task_(\d+)\.csv$")
    max_index = 0
    for p in sim_dir.glob("task_*.csv"):
        m = pattern.match(p.name)
        if m:
            max_index = max(max_index, int(m.group(1)))
    next_index = max_index + 1
    return sim_dir / f"task_{next_index}.csv", next_index


class SingleJointRangeLimiter:
    def __init__(self, robot, joint_name, hold_effort=5000.0, limit_rad=np.deg2rad(1.0)):
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.dof_names = list(robot.dof_names)
        if joint_name not in self.dof_names:
            raise RuntimeError(f"Requested limit joint '{joint_name}' was not found in robot DOFs: {self.dof_names}")
        self.joint_name = str(joint_name)
        self.joint_index = int(self.dof_names.index(self.joint_name))
        self.hold_effort = float(hold_effort)
        self.limit_rad = abs(float(limit_rad))
        self.enabled = False
        self.center_rad = None

    def configure_drive(self):
        try:
            self.robot._articulation_view.set_max_efforts(
                values=np.array([self.hold_effort], dtype=np.float32),
                joint_indices=np.array([self.joint_index], dtype=np.int32),
            )
        except Exception:
            pass

    def set_center_rad(self, center_rad):
        self.center_rad = float(center_rad)

    def get_limits_rad(self):
        if self.center_rad is None:
            raise RuntimeError(f"Range center for {self.joint_name} has not been set.")
        return self.center_rad - self.limit_rad, self.center_rad + self.limit_rad

    def clamp_value(self, value_rad):
        lower, upper = self.get_limits_rad()
        return float(np.clip(float(value_rad), lower, upper))

    def enable(self, center_rad=None, limit_rad=None):
        if center_rad is not None:
            self.set_center_rad(center_rad)
        if limit_rad is not None:
            self.limit_rad = abs(float(limit_rad))
        if self.center_rad is None:
            raise RuntimeError(f"Range center for {self.joint_name} has not been set.")
        self.enabled = True
        self.configure_drive()
        self.hold_step()

    def disable(self):
        self.enabled = False

    def hold_step(self):
        if not self.enabled or self.center_rad is None:
            return
        js = self.robot.get_joints_state()
        if js is None or js.positions is None:
            return
        current = float(np.array(js.positions, dtype=np.float32)[self.joint_index])
        clamped = self.clamp_value(current)
        if abs(clamped - current) < 1e-6:
            return
        action = ArticulationAction(
            joint_positions=np.array([clamped], dtype=np.float32),
            joint_indices=np.array([self.joint_index], dtype=np.int32),
        )
        self.controller.apply_action(action)

    def apply_to_arm_action(self, action, idx_list):
        if not self.enabled or self.center_rad is None:
            return action
        if action is None or idx_list is None or self.joint_index not in idx_list:
            return action
        idx_list_arr = np.asarray(idx_list)
        local_idx = int(np.where(idx_list_arr == self.joint_index)[0][0])
        pos = np.array(action.joint_positions, dtype=np.float32, copy=True)
        vel = None if action.joint_velocities is None else np.array(action.joint_velocities, dtype=np.float32, copy=True)
        original = float(pos[local_idx])
        clamped = self.clamp_value(original)
        pos[local_idx] = clamped
        if vel is not None and local_idx < len(vel) and abs(clamped - original) > 1e-6:
            vel[local_idx] = 0.0
        return ArticulationAction(
            joint_positions=pos,
            joint_velocities=vel,
            joint_indices=action.joint_indices,
        )

    def clamp_joint_array_in_place(self, joint_positions, joint_names):
        if (not self.enabled) or (self.center_rad is None):
            return joint_positions
        if self.joint_name not in joint_names:
            return joint_positions
        local_idx = joint_names.index(self.joint_name)
        joint_positions[local_idx] = self.clamp_value(joint_positions[local_idx])
        return joint_positions


class PiperGripperAdapter:
    def __init__(self, robot, hold_effort=2000.0, hold_kp=None, hold_kd=None):
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.dof_names = list(robot.dof_names)
        self.joint_names = ["joint7", "joint8"]
        self.joint_indices = [self.dof_names.index(n) for n in self.joint_names]
        self.opened_positions = np.array([0.035, -0.035], dtype=np.float32)
        self.closed_positions = np.array([0.0, 0.0], dtype=np.float32)
        self.hold_enabled = False
        self.hold_target = self.closed_positions.copy()
        self.hold_effort = float(hold_effort)
        self.hold_kp = None if hold_kp is None else float(hold_kp)
        self.hold_kd = None if hold_kd is None else float(hold_kd)

    def configure_hold_drive(self):
        try:
            self.robot._articulation_view.set_max_efforts(
                values=np.array([self.hold_effort for _ in self.joint_indices], dtype=np.float32),
                joint_indices=self.joint_indices,
            )
        except Exception:
            pass
        if self.hold_kp is None and self.hold_kd is None:
            return
        try:
            kps = np.array([self.hold_kp for _ in self.joint_indices], dtype=np.float32)
            kds = np.array([self.hold_kd for _ in self.joint_indices], dtype=np.float32)
            self.robot._articulation_view.set_gains(kps=kps, kds=kds, joint_indices=self.joint_indices)
        except Exception:
            pass

    def apply_positions(self, q):
        action = ArticulationAction(
            joint_positions=np.array(q, dtype=np.float32),
            joint_indices=self.joint_indices,
        )
        self.controller.apply_action(action)

    def open(self):
        self.hold_enabled = False
        self.apply_positions(self.opened_positions)

    def close(self):
        self.apply_positions(self.closed_positions)

    def enable_hold(self, q=None):
        if q is None:
            q = self.closed_positions
        self.hold_target = np.array(q, dtype=np.float32)
        self.hold_enabled = True
        self.configure_hold_drive()
        self.apply_positions(self.hold_target)

    def disable_hold(self):
        self.hold_enabled = False

    def hold_step(self):
        if self.hold_enabled:
            self.apply_positions(self.hold_target)


class PickPlaceState:
    SNAPSHOT_OBJECT = "SNAPSHOT_OBJECT"
    MOVE_PRE_PICK = "MOVE_PRE_PICK"  # pre-pick transport and grasp-yaw alignment run together
    MOVE_PICK = "MOVE_PICK"
    CLOSE_AND_LIFT = "CLOSE_AND_LIFT"  # close the gripper and lift in the same phase
    VERIFY_GRASP = "VERIFY_GRASP"
    REORIENT_FOR_TRANSPORT = "REORIENT_FOR_TRANSPORT"
    MOVE_PRE_PLACE = "MOVE_PRE_PLACE"
    MOVE_PLACE = "MOVE_PLACE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    RETREAT = "RETREAT"
    MOVE_HOME = "MOVE_HOME"
    DONE = "DONE"
    FAILED = "FAILED"


def wrap_to_pi(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def canonicalize_mod_pi(angle):
    return ((float(angle) + (0.5 * np.pi)) % np.pi) - (0.5 * np.pi)


def choose_nearest_half_turn_equivalent(angle_mod_pi, reference):
    base = canonicalize_mod_pi(angle_mod_pi)
    candidates = [base + k * np.pi for k in range(-3, 4)]
    return min(candidates, key=lambda a: abs(float(a) - float(reference)))


def quat_wxyz_from_yaw(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def yaw_from_quat_wxyz(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_plan(motion_gen, tensor_args, plan_config, robot, sim_js, goal_position, goal_quat):
    sim_js_names = robot.dof_names
    cu_js = JointState(
        position=tensor_args.to_device(sim_js.positions),
        velocity=tensor_args.to_device(sim_js.velocities),
        acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
        jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
        joint_names=sim_js_names,
    )
    cu_js.velocity *= 0.0
    cu_js.acceleration *= 0.0
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    ik_goal = Pose(
        position=tensor_args.to_device(goal_position),
        quaternion=tensor_args.to_device(goal_quat),
    )
    result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
    if not result.success.item():
        carb.log_warn(f"Plan failed: {result.status}")
        return None, None

    cmd_plan = result.get_interpolated_plan()
    cmd_plan = motion_gen.get_full_js(cmd_plan)

    idx_list = []
    common_js_names = []
    for name in sim_js_names:
        if name in cmd_plan.joint_names:
            idx_list.append(robot.get_dof_index(name))
            common_js_names.append(name)
    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
    return cmd_plan, idx_list


def build_joint_plan(
    motion_gen,
    tensor_args,
    plan_config,
    robot,
    sim_js,
    goal_joint_positions,
    goal_joint_names,
):
    """Plan collision-aware motion to an exact joint configuration with cuRobo."""
    sim_js_names = list(robot.dof_names)

    start_js = JointState(
        position=tensor_args.to_device(np.asarray(sim_js.positions, dtype=np.float32)),
        velocity=tensor_args.to_device(np.asarray(sim_js.velocities, dtype=np.float32)),
        acceleration=tensor_args.to_device(np.asarray(sim_js.velocities, dtype=np.float32)) * 0.0,
        jerk=tensor_args.to_device(np.asarray(sim_js.velocities, dtype=np.float32)) * 0.0,
        joint_names=sim_js_names,
    )
    start_js.velocity *= 0.0
    start_js.acceleration *= 0.0
    start_js = start_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    goal_np = np.asarray(goal_joint_positions, dtype=np.float32)
    goal_pos = tensor_args.to_device(goal_np)
    goal_state = JointState(
        position=goal_pos,
        velocity=goal_pos * 0.0,
        acceleration=goal_pos * 0.0,
        jerk=goal_pos * 0.0,
        joint_names=list(goal_joint_names),
    )
    goal_state = goal_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)

    result = motion_gen.plan_single_js(
        start_js.unsqueeze(0),
        goal_state.unsqueeze(0),
        plan_config,
    )

    if not result.success.item():
        carb.log_warn(f"Joint-space plan failed: {result.status}")
        return None, None

    cmd_plan = result.get_interpolated_plan()
    cmd_plan = motion_gen.get_full_js(cmd_plan)

    idx_list = []
    common_js_names = []
    for name in sim_js_names:
        if name in cmd_plan.joint_names:
            idx_list.append(robot.get_dof_index(name))
            common_js_names.append(name)

    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
    return cmd_plan, idx_list


def get_ee_prim(robot_prim_path, ee_link_name="gripper_base"):
    # robot_prim_path is the Piper container, e.g. /World/robot_testbed/piper.
    # Do not truncate it to /World/<name>; the testbed introduces an extra hierarchy level.
    return XFormPrim(f"{robot_prim_path}/{ee_link_name}")


def get_arm_velocities(robot, sim_js, joint_names):
    return np.array(
        [sim_js.velocities[robot.get_dof_index(name)] for name in joint_names],
        dtype=np.float32,
    )


def is_arm_static(robot, sim_js, joint_names, threshold):
    arm_vel = get_arm_velocities(robot, sim_js, joint_names)
    return bool(np.max(np.abs(arm_vel)) < threshold), arm_vel


def settle_after_gripper(my_world, robot, joint_names, threshold, max_steps, gripper=None, joint6_limiter=None):
    for _ in range(max_steps):
        if gripper is not None and gripper.hold_enabled and not args.disable_continuous_gripper_hold:
            gripper.hold_step()
        if joint6_limiter is not None and joint6_limiter.enabled:
            joint6_limiter.hold_step()
        sim_js_tmp = robot.get_joints_state()
        if sim_js_tmp is None or np.any(np.isnan(sim_js_tmp.velocities)):
            my_world.step(render=True)
            continue
        arm_static, arm_vel = is_arm_static(robot, sim_js_tmp, joint_names, threshold)
        if arm_static:
            return True, arm_vel
        my_world.step(render=True)
    sim_js_tmp = robot.get_joints_state()
    if sim_js_tmp is None or np.any(np.isnan(sim_js_tmp.velocities)):
        return False, None
    _, arm_vel = is_arm_static(robot, sim_js_tmp, joint_names, threshold)
    return False, arm_vel


def set_arm_max_efforts(robot, arm_idx_list, effort):
    try:
        robot._articulation_view.set_max_efforts(
            values=np.array([float(effort) for _ in range(len(arm_idx_list))], dtype=np.float32),
            joint_indices=arm_idx_list,
        )
    except Exception:
        pass


def configure_arm_drive_gains(robot, arm_idx_list):
    """Match the joint1..joint6 drive gains used by the legacy helper.py URDF importer."""
    idx = np.array(arm_idx_list, dtype=np.int32)
    kps = np.full(len(idx), float(args.arm_drive_kp), dtype=np.float32)
    kds = np.full(len(idx), float(args.arm_drive_kd), dtype=np.float32)

    try:
        robot._articulation_view.set_gains(
            kps=kps,
            kds=kds,
            joint_indices=idx,
        )
        print(
            f'[DRIVE] Applied legacy-compatible arm gains: '
            f'kp={args.arm_drive_kp:.5f}, kd={args.arm_drive_kd:.5f}, '
            f'joints={len(idx)}'
        )
        return True
    except Exception as exc:
        carb.log_warn(f'Failed to apply arm drive gains: {exc}')
        return False


def reset_episode(robot, gripper, default_config, arm_idx_list, joint6_limiter=None):
    try:
        robot._articulation_view.initialize()
    except Exception:
        pass
    robot.set_joint_positions(default_config, arm_idx_list)
    robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
    if joint6_limiter is not None:
        joint6_limiter.disable()
    if gripper is not None:
        gripper.disable_hold()
        gripper.open()


def compute_top_grasp_plan(object_pos, object_yaw_rad, place_pos, place_yaw_rad):
    # Axis mapping fix:
    # For top_grasp_short_axis, the jaws should close across the object's short axis
    # without the previous extra +90 deg rotation. The long-axis grasp needs the +90 deg offset.
    direction_mode_offset_deg = 0.0 if args.grasp_direction_mode == "top_grasp_short_axis" else 90.0
    desired_object_jaw_world_yaw = canonicalize_mod_pi(
        float(object_yaw_rad) + np.deg2rad(direction_mode_offset_deg + args.grasp_yaw_trim_deg)
    )
    desired_place_jaw_world_yaw = canonicalize_mod_pi(
        float(place_yaw_rad) + np.deg2rad(direction_mode_offset_deg + args.grasp_yaw_trim_deg)
    )

    top_face_z = float(object_pos[2]) + (0.5 * float(args.object_short_size))
    place_top_face_z = float(place_pos[2]) + (0.5 * float(args.object_short_size))

    pick = np.array(object_pos, dtype=np.float32).copy()
    pick[2] = top_face_z + float(args.gripper_base_above_top)

    pre_pick = np.array(object_pos, dtype=np.float32).copy()
    pre_pick[2] = max(float(object_pos[2]) + float(args.approach_height), float(pick[2]) + 0.04)

    lift = pick.copy()
    lift[2] += float(args.lift_delta)

    place = np.array(place_pos, dtype=np.float32).copy()
    place[2] = place_top_face_z + float(args.gripper_base_above_top)

    pre_place = np.array(place_pos, dtype=np.float32).copy()
    pre_place[2] = max(float(place_pos[2]) + float(args.approach_height), float(place[2]) + 0.04)
    retreat = pre_place.copy()

    transport_joint6_target = np.deg2rad(args.transport_joint6_deg)

    return {
        "goals": {
            PickPlaceState.MOVE_PRE_PICK: pre_pick,
            PickPlaceState.MOVE_PICK: pick,
            PickPlaceState.CLOSE_AND_LIFT: lift,
            PickPlaceState.MOVE_PRE_PLACE: pre_place,
            PickPlaceState.MOVE_PLACE: place,
            PickPlaceState.RETREAT: retreat,
        },
        "shape_info": {
            "object_top_face_z": top_face_z,
            "place_top_face_z": place_top_face_z,
            "pick_target_z": float(pick[2]),
            "place_target_z": float(place[2]),
        },
        "desired_object_jaw_world_yaw_rad": desired_object_jaw_world_yaw,
        "desired_place_jaw_world_yaw_rad": desired_place_jaw_world_yaw,
        "transport_joint6_target_rad": transport_joint6_target,
    }

def get_object_and_ee_pose(pick_object, ee_prim):
    obj_pos, _ = pick_object.get_world_pose()
    ee_pos, _ = ee_prim.get_world_pose()
    return np.array(obj_pos, dtype=np.float32), np.array(ee_pos, dtype=np.float32)


def get_ee_pose(ee_prim):
    ee_pos, ee_quat = ee_prim.get_world_pose()
    return np.array(ee_pos, dtype=np.float32), np.array(ee_quat, dtype=np.float32)


def get_current_jaw_world_yaw(ee_prim):
    _, ee_quat = get_ee_pose(ee_prim)
    ee_yaw = yaw_from_quat_wxyz(ee_quat)
    jaw_yaw = wrap_to_pi(ee_yaw + np.deg2rad(args.ee_jaw_yaw_offset_deg))
    return jaw_yaw, ee_yaw, ee_quat


def choose_grasp_world_yaw_branch(desired_jaw_world_yaw_rad, current_jaw_world_yaw_rad):
    return float(
        choose_nearest_half_turn_equivalent(
            desired_jaw_world_yaw_rad,
            current_jaw_world_yaw_rad,
        )
    )


def solve_joint6_target_for_world_yaw(robot, ee_prim, locked_target_world_yaw_rad):
    current_joint6 = get_joint_position(robot, "joint6")
    if current_joint6 is None:
        return None

    current_jaw_world_yaw, current_ee_yaw, current_ee_quat = get_current_jaw_world_yaw(ee_prim)
    world_yaw_error = wrap_to_pi(float(locked_target_world_yaw_rad) - current_jaw_world_yaw)
    joint_delta = float(args.joint6_world_yaw_sign) * world_yaw_error
    target_joint6 = current_joint6 + joint_delta

    return {
        "current_joint6_rad": float(current_joint6),
        "current_jaw_world_yaw_rad": float(current_jaw_world_yaw),
        "current_ee_yaw_rad": float(current_ee_yaw),
        "current_ee_quat_wxyz": np.array(current_ee_quat, dtype=np.float32),
        "desired_target_world_yaw_rad": float(locked_target_world_yaw_rad),
        "world_yaw_error_rad": float(world_yaw_error),
        "joint_delta_rad": float(joint_delta),
        "target_joint6_rad": float(target_joint6),
    }


def solve_dynamic_joint6_target(robot, ee_prim, desired_jaw_world_yaw_rad):
    current_jaw_world_yaw, _, _ = get_current_jaw_world_yaw(ee_prim)
    locked_target_world_yaw_rad = choose_grasp_world_yaw_branch(
        desired_jaw_world_yaw_rad,
        current_jaw_world_yaw,
    )
    return solve_joint6_target_for_world_yaw(robot, ee_prim, locked_target_world_yaw_rad)


def evaluate_grasp_success(pick_object, ee_prim, close_object_pos, close_rel_offset):
    obj_pos, ee_pos = get_object_and_ee_pose(pick_object, ee_prim)
    expected_obj_pos = np.array(ee_pos, dtype=np.float32) + np.array(close_rel_offset, dtype=np.float32)
    lift_amount = float(obj_pos[2] - close_object_pos[2])
    carry_error = float(np.linalg.norm(obj_pos - expected_obj_pos))
    success = (lift_amount >= args.grasp_min_lift) and (carry_error <= args.grasp_max_ee_error)
    return success, {
        "object_pos": obj_pos,
        "ee_pos": ee_pos,
        "lift_amount": lift_amount,
        "carry_error": carry_error,
        "expected_object_pos": expected_obj_pos,
    }


def is_object_still_carried(pick_object, ee_prim, close_rel_offset):
    obj_pos, ee_pos = get_object_and_ee_pose(pick_object, ee_prim)
    expected_obj_pos = np.array(ee_pos, dtype=np.float32) + np.array(close_rel_offset, dtype=np.float32)
    carry_error = float(np.linalg.norm(obj_pos - expected_obj_pos))
    carried = carry_error <= max(args.grasp_max_ee_error, 0.12)
    return carried, obj_pos, ee_pos, carry_error


def report_final_place_quality(pick_object, place_marker):
    obj_pos, _ = pick_object.get_world_pose()
    place_pos, _ = place_marker.get_world_pose()
    obj_pos = np.array(obj_pos, dtype=np.float32)
    place_pos = np.array(place_pos, dtype=np.float32)
    place_pos[2] = float(args.place_xyz[2] if args.place_goal_fixed_z is None else args.place_goal_fixed_z)
    xy_error = float(np.linalg.norm((obj_pos - place_pos)[:2]))
    z_error = float(abs(obj_pos[2] - place_pos[2]))
    success = xy_error <= args.release_max_place_error
    return success, obj_pos, place_pos, xy_error, z_error


def get_joint_position(robot, joint_name):
    js = robot.get_joints_state()
    if js is None:
        return None
    dof_names = list(robot.dof_names)
    if joint_name not in dof_names:
        return None
    idx = dof_names.index(joint_name)
    return float(np.array(js.positions, dtype=np.float32)[idx])


def drive_joint_toward(robot, joint_name, target_rad, max_step_rad):
    dof_names = list(robot.dof_names)
    if joint_name not in dof_names:
        raise RuntimeError(f"Joint {joint_name} not found in robot DOFs: {dof_names}")
    joint_index = dof_names.index(joint_name)
    js = robot.get_joints_state()
    if js is None or js.positions is None:
        return False, None, None
    current = float(np.array(js.positions, dtype=np.float32)[joint_index])
    error = wrap_to_pi(target_rad - current)
    if abs(error) <= np.deg2rad(args.joint6_align_tolerance_deg):
        cmd = target_rad
        done = True
    else:
        step = float(np.clip(error, -max_step_rad, max_step_rad))
        cmd = current + step
        done = False
    controller = robot.get_articulation_controller()
    action = ArticulationAction(
        joint_positions=np.array([cmd], dtype=np.float32),
        joint_indices=np.array([joint_index], dtype=np.int32),
    )
    controller.apply_action(action)
    return done, current, cmd


def drive_arm_joints_toward(robot, joint_names, target_positions_rad, max_step_rad, tolerance_rad):
    js = robot.get_joints_state()
    if js is None or js.positions is None:
        return False, None, None
    dof_names = list(robot.dof_names)
    joint_indices = np.array([dof_names.index(name) for name in joint_names], dtype=np.int32)
    current = np.array([js.positions[idx] for idx in joint_indices], dtype=np.float32)
    target = np.array(target_positions_rad, dtype=np.float32)
    error = target - current
    done = bool(np.all(np.abs(error) <= tolerance_rad))
    if done:
        cmd = target.copy()
    else:
        step = np.clip(error, -max_step_rad, max_step_rad)
        cmd = current + step
    action = ArticulationAction(
        joint_positions=np.array(cmd, dtype=np.float32),
        joint_indices=joint_indices,
    )
    robot.get_articulation_controller().apply_action(action)
    return done, current, cmd


def sample_random_object_pose(rng, spawn_center_xy, spawn_size_xy, spawn_z):
    x = float(spawn_center_xy[0]) + (rng.random() - 0.5) * float(spawn_size_xy[0])
    y = float(spawn_center_xy[1]) + (rng.random() - 0.5) * float(spawn_size_xy[1])
    yaw = rng.uniform(-math.pi, math.pi)
    pos = np.array([x, y, float(spawn_z)], dtype=np.float32)
    quat = quat_wxyz_from_yaw(yaw)
    return pos, quat, yaw


def respawn_object(pick_object, pos, quat):
    pick_object.set_world_pose(position=pos, orientation=quat)
    pick_object.set_linear_velocity(np.zeros(3, dtype=np.float32))
    pick_object.set_angular_velocity(np.zeros(3, dtype=np.float32))


def create_spawn_region_visuals(spawn_center_xy, spawn_size_xy, spawn_z, object_short_size):
    visuals = []
    if not args.show_spawn_region:
        return visuals
    thickness = 0.003
    border_height = 0.002
    z = float(spawn_z) - 0.5 * float(object_short_size) + 0.5 * border_height
    cx = float(spawn_center_xy[0])
    cy = float(spawn_center_xy[1])
    sx = float(spawn_size_xy[0])
    sy = float(spawn_size_xy[1])
    try:
        visuals.append(cuboid.VisualCuboid('/World/spawn_region_north', position=np.array([cx, cy + 0.5 * sy, z], dtype=np.float32), orientation=np.array([1.0,0.0,0.0,0.0], dtype=np.float32), color=np.array([0.8,0.8,0.2]), size=1.0, scale=np.array([sx, thickness, border_height], dtype=np.float32)))
        visuals.append(cuboid.VisualCuboid('/World/spawn_region_south', position=np.array([cx, cy - 0.5 * sy, z], dtype=np.float32), orientation=np.array([1.0,0.0,0.0,0.0], dtype=np.float32), color=np.array([0.8,0.8,0.2]), size=1.0, scale=np.array([sx, thickness, border_height], dtype=np.float32)))
        visuals.append(cuboid.VisualCuboid('/World/spawn_region_east', position=np.array([cx + 0.5 * sx, cy, z], dtype=np.float32), orientation=np.array([1.0,0.0,0.0,0.0], dtype=np.float32), color=np.array([0.8,0.8,0.2]), size=1.0, scale=np.array([thickness, sy, border_height], dtype=np.float32)))
        visuals.append(cuboid.VisualCuboid('/World/spawn_region_west', position=np.array([cx - 0.5 * sx, cy, z], dtype=np.float32), orientation=np.array([1.0,0.0,0.0,0.0], dtype=np.float32), color=np.array([0.8,0.8,0.2]), size=1.0, scale=np.array([thickness, sy, border_height], dtype=np.float32)))
    except Exception as exc:
        carb.log_warn(f"Spawn-region visual guide could not be created: {exc}")
    return visuals




def _set_pose_xyz_in_place(obj, xyz):
    try:
        pose = list(obj.pose)
        pose[0] = float(xyz[0])
        pose[1] = float(xyz[1])
        pose[2] = float(xyz[2])
        obj.pose = pose
        return
    except Exception:
        pass
    try:
        obj.pose[0] = float(xyz[0])
        obj.pose[1] = float(xyz[1])
        obj.pose[2] = float(xyz[2])
    except Exception:
        pass


def _set_absolute_size_xyz_in_place(obj, size_xyz):
    size_xyz = np.array(size_xyz, dtype=np.float32)
    try:
        if hasattr(obj, 'dims') and obj.dims is not None:
            dims = np.array(obj.dims, dtype=np.float32)
            if dims.shape[0] >= 3:
                dims[:3] = size_xyz[:3]
                obj.dims = dims.tolist()
                return
    except Exception:
        pass
    try:
        if hasattr(obj, 'scale') and obj.scale is not None:
            scale = np.array(obj.scale, dtype=np.float32)
            if scale.shape[0] >= 3:
                scale[:3] = size_xyz[:3]
                obj.scale = scale.tolist()
                return
    except Exception:
        pass


def _set_translate_op(stage, prim_path, xyz):
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return True


def _set_scale_op(stage, prim_path, xyz):
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return True


def _quat_wxyz_from_euler_xyz_deg(rpy_deg):
    rx = math.radians(float(rpy_deg[0])) * 0.5
    ry = math.radians(float(rpy_deg[1])) * 0.5
    rz = math.radians(float(rpy_deg[2])) * 0.5

    qx = np.array([math.cos(rx), math.sin(rx), 0.0, 0.0], dtype=np.float64)
    qy = np.array([math.cos(ry), 0.0, math.sin(ry), 0.0], dtype=np.float64)
    qz = np.array([math.cos(rz), 0.0, 0.0, math.sin(rz)], dtype=np.float64)

    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    q = _quat_mul(_quat_mul(qx, qy), qz)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q


def _set_translate_orient_exact(stage, prim_path, xyz, rpy_deg):
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    try:
        xformable.ClearXformOpOrder()
    except Exception:
        pass

    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()

    translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    qw, qx, qy, qz = _quat_wxyz_from_euler_xyz_deg(rpy_deg)
    orient_op.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))

    try:
        xformable.SetXformOpOrder([translate_op, orient_op])
    except Exception:
        pass
    return True


def _normalize_prim_path(path):
    path = str(path).strip()
    if not path.startswith('/'):
        path = '/' + path
    return path.rstrip('/')


def _assert_testbed_prim(stage, path, label):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Testbed {label} prim was not found at {path}. "
            "Check the testbed defaultPrim/reference composition and --testbed_prim_path."
        )
    return prim


def load_testbed_and_wrap_robot(stage, my_world, base_link_name, ee_link_name):
    testbed_usd = str(Path(args.testbed_usd_path).expanduser().resolve())
    if not Path(testbed_usd).is_file():
        raise FileNotFoundError(f"Testbed USD not found: {testbed_usd}")

    testbed_prim_path = _normalize_prim_path(args.testbed_prim_path)
    testbed_prim = stage.DefinePrim(testbed_prim_path, 'Xform')
    refs = testbed_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(testbed_usd)

    robot_prim_path = f'{testbed_prim_path}/piper'
    base_link_path = f'{robot_prim_path}/{base_link_name}'
    ee_link_path = f'{robot_prim_path}/{ee_link_name}'
    articulation_root_path = f'{robot_prim_path}/root_joint'
    table_prim_path = f'{testbed_prim_path}/Table'
    mount_prim_path = f'{table_prim_path}/Frames/robot_mount'

    _assert_testbed_prim(stage, robot_prim_path, 'Piper container')
    _assert_testbed_prim(stage, base_link_path, 'base link')
    _assert_testbed_prim(stage, ee_link_path, 'end-effector link')
    _assert_testbed_prim(stage, table_prim_path, 'Table')
    cad_correction_path = f'{table_prim_path}/CAD_Correction'
    if not stage.GetPrimAtPath(cad_correction_path).IsValid():
        carb.log_warn(
            f'Table CAD hierarchy was not found at {cad_correction_path}. '
            'The table_asset.usd reference inside the testbed may be unresolved. '
            'The simplified tabletop collision proxy can still be created, but verify the visual asset reference.'
        )

    articulation_root = _assert_testbed_prim(stage, articulation_root_path, 'articulation root')
    if not articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        carb.log_warn(
            f'Expected ArticulationRootAPI at {articulation_root_path}, but the API is not present.'
        )

    robot = Robot(prim_path=base_link_path, name='robot')
    robot = my_world.scene.add(robot)

    # Match the initialization behavior of the old helper.add_robot_to_scene() path,
    # but do not import a second URDF robot.
    try:
        my_world.initialize_physics()
        robot.initialize()
    except Exception as exc:
        carb.log_warn(f'Initial testbed robot initialization was deferred: {exc}')

    # Mount/base alignment sanity check. The testbed was authored so both frames coincide.
    mount_prim = stage.GetPrimAtPath(mount_prim_path)
    if mount_prim.IsValid():
        try:
            xform_cache = UsdGeom.XformCache()
            base_pos = np.array(
                xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(base_link_path)).ExtractTranslation(),
                dtype=np.float64,
            )
            mount_pos = np.array(
                xform_cache.GetLocalToWorldTransform(mount_prim).ExtractTranslation(),
                dtype=np.float64,
            )
            mount_error = float(np.linalg.norm(base_pos - mount_pos))
            print(
                f'[TESTBED] base_link={base_pos.tolist()} robot_mount={mount_pos.tolist()} '
                f'position_error={mount_error:.6e} m'
            )
            if mount_error > 1.0e-3:
                carb.log_warn(
                    f'Piper base_link and table robot_mount differ by {mount_error:.6f} m. '
                    'Recheck the testbed assembly before collecting data.'
                )
        except Exception as exc:
            carb.log_warn(f'Could not verify base_link/robot_mount alignment: {exc}')

    print(f'[TESTBED] Referenced {testbed_usd} at {testbed_prim_path}')
    print(f'[TESTBED] Piper container: {robot_prim_path}')
    print(f'[TESTBED] Robot wrapper:   {base_link_path}')
    print(f'[TESTBED] EE link:         {ee_link_path}')

    return robot, robot_prim_path, table_prim_path


def create_tabletop_collision_proxy(stage, table_prim_path):
    if args.disable_tabletop_collision_proxy:
        print('[TESTBED] Simplified tabletop collision proxy disabled by flag.')
        return None

    size_xyz = np.array(args.tabletop_proxy_size_xyz, dtype=np.float32)
    local_xyz = np.array(args.tabletop_proxy_local_xyz, dtype=np.float32)
    proxy_parent_path = f'{table_prim_path}/CollisionProxies'
    proxy_path = f'{proxy_parent_path}/tabletop'

    stage.DefinePrim(proxy_parent_path, 'Xform')
    cube = UsdGeom.Cube.Define(stage, proxy_path)
    cube.CreateSizeAttr(1.0)
    _set_translate_op(stage, proxy_path, local_xyz)
    _set_scale_op(stage, proxy_path, size_xyz)

    try:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    except Exception as exc:
        carb.log_warn(f'Failed to apply CollisionAPI to tabletop proxy: {exc}')

    # Keep the proxy physically active while hiding it from normal rendering.
    # USD visibility does not remove the PhysX CollisionAPI from the prim.
    try:
        imageable = UsdGeom.Imageable(cube.GetPrim())
        if args.show_tabletop_collision_proxy:
            imageable.MakeVisible()
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])
        else:
            imageable.MakeInvisible()
    except Exception as exc:
        carb.log_warn(f'Could not update tabletop proxy visibility: {exc}')

    print(
        f'[TESTBED] Tabletop collision proxy: {proxy_path} '
        f'local_xyz={local_xyz.tolist()} size={size_xyz.tolist()}'
    )
    return proxy_path


def build_initial_table_world_config(stage, tabletop_proxy_path, table_prim_path):
    """Build the initial cuRobo tabletop world entirely in memory.

    The tabletop geometry already lives in this project, so there is no need
    to read an installed cuRobo sample world file just to obtain a cuboid
    template and overwrite it.
    """
    size_xyz = np.array(args.tabletop_proxy_size_xyz, dtype=np.float32)
    local_xyz = np.array(args.tabletop_proxy_local_xyz, dtype=np.float32)
    center_xyz = local_xyz.copy()

    if tabletop_proxy_path is not None:
        proxy_prim = stage.GetPrimAtPath(tabletop_proxy_path)
        if proxy_prim.IsValid():
            try:
                xform_cache = UsdGeom.XformCache()
                center_xyz = np.array(
                    xform_cache.GetLocalToWorldTransform(proxy_prim).ExtractTranslation(),
                    dtype=np.float32,
                )
            except Exception as exc:
                carb.log_warn(
                    f'Failed to query tabletop proxy world pose; '
                    f'using Table transform fallback: {exc}'
                )
                proxy_prim = None
        else:
            proxy_prim = None
    else:
        proxy_prim = None

    if proxy_prim is None:
        table_prim = stage.GetPrimAtPath(table_prim_path)
        if table_prim.IsValid():
            try:
                xform_cache = UsdGeom.XformCache()
                table_world = xform_cache.GetLocalToWorldTransform(table_prim)
                table_origin = np.array(
                    table_world.ExtractTranslation(),
                    dtype=np.float32,
                )
                # The validated testbed keeps Table rotation aligned with the
                # Piper/world frame.
                center_xyz = table_origin + local_xyz
            except Exception as exc:
                carb.log_warn(
                    f'Failed to query Table world pose for cuRobo tabletop '
                    f'cuboid: {exc}'
                )

    tabletop_dict = {
        'cuboid': {
            'tabletop_proxy': {
                'dims': [float(v) for v in size_xyz],
                'pose': [
                    float(center_xyz[0]),
                    float(center_xyz[1]),
                    float(center_xyz[2]),
                    1.0, 0.0, 0.0, 0.0,
                ],
            }
        }
    }

    world_cfg_table = WorldConfig.from_dict(tabletop_dict)

    print(
        f'[CUROBO] Initial tabletop cuboid center={center_xyz.tolist()} '
        f'size={size_xyz.tolist()} (in-memory project config)'
    )
    return WorldConfig(cuboid=world_cfg_table.cuboid, mesh=[])



def create_workspace_lights(stage):
    if args.disable_scene_lights:
        print('[LIGHT] Workspace lights disabled by flag.')
        return []

    root_path = '/World/Lights'
    stage.DefinePrim(root_path, 'Xform')

    positions = [
        ( 2.0,  2.0, 2.0),
        (-2.0,  2.0, 2.0),
        ( 2.0, -2.0, 2.0),
        (-2.0, -2.0, 2.0),
    ]

    created = []
    for i, xyz in enumerate(positions, start=1):
        path = f'{root_path}/workspace_light_{i}'
        light = UsdLux.SphereLight.Define(stage, path)
        try:
            light.CreateIntensityAttr(float(args.scene_light_intensity))
            light.CreateRadiusAttr(float(args.scene_light_radius))
            light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        except Exception:
            pass
        _set_translate_op(stage, path, np.array(xyz, dtype=np.float32))
        created.append(path)

    print(
        f'[LIGHT] Created {len(created)} SphereLights at '
        f'(±2, ±2, 2), intensity={args.scene_light_intensity:.1f}, '
        f'radius={args.scene_light_radius:.3f} m'
    )
    return created


def add_ground_plane(my_world):
    try:
        my_world.scene.add_default_ground_plane(z_position=float(args.ground_z))
    except TypeError:
        gp = my_world.scene.add_default_ground_plane()
        try:
            gp.set_world_pose(
                position=np.array([0.0, 0.0, float(args.ground_z)], dtype=np.float32)
            )
        except Exception:
            pass


def _iter_mesh_prims(root_prim):
    if not root_prim or not root_prim.IsValid():
        return []
    return [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)]


def create_basket_reference_with_collision(stage):
    basket_path = '/World/basket'
    basket_xyz = np.array(args.basket_xyz, dtype=np.float32)
    basket_prim = stage.DefinePrim(basket_path, 'Xform')
    refs = basket_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(str(args.basket_usd_path))
    _set_translate_op(stage, basket_path, basket_xyz)

    mesh_count = 0
    for mesh_prim in _iter_mesh_prims(basket_prim):
        try:
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
        except Exception:
            pass
        try:
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_api.CreateApproximationAttr().Set(args.basket_collision_approximation)
        except Exception:
            pass
        mesh_count += 1

    if mesh_count == 0:
        carb.log_warn(f"Basket USD was referenced at {basket_path}, but no mesh prims were found under it for collision setup.")
    else:
        carb.log_info(
            f"Basket USD loaded at {basket_path} with {mesh_count} mesh collider prim(s); approximation={args.basket_collision_approximation}."
        )

    return XFormPrim(basket_path)


def create_place_marker_xform(stage):
    place_marker_path = '/World/place_marker'
    stage.DefinePrim(place_marker_path, 'Xform')
    _set_translate_op(stage, place_marker_path, np.array(args.place_marker_xyz, dtype=np.float32))
    return XFormPrim(place_marker_path)


def _create_preview_surface_material(stage, material_path, rgb, roughness=0.4, metallic=0.0):
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _apply_display_color_fallback(prim, rgb):
    try:
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])
    except Exception:
        pass


def _bind_material_to_prim(stage, prim_path, material, rgb_fallback=None):
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        carb.log_warn(f"Material target prim not found: {prim_path}")
        return False
    try:
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        if rgb_fallback is not None:
            _apply_display_color_fallback(prim, rgb_fallback)
        return True
    except Exception as exc:
        carb.log_warn(f"Failed to bind material to {prim_path}: {exc}")
        return False


def apply_basket_material(stage):
    """Apply only the basket material; keep the robot's description/USD materials untouched."""
    looks_root = "/World/Looks"
    stage.DefinePrim(looks_root, "Scope")

    basket_rgb = (1.0, 0.9, 0.05)
    basket_material = _create_preview_surface_material(
        stage,
        f"{looks_root}/BasketYellow",
        rgb=basket_rgb,
        roughness=0.45,
        metallic=0.0,
    )

    basket_targets = [
        "/World/basket/basket/basket",
    ]

    bound_count = 0
    for p in basket_targets:
        bound_count += int(_bind_material_to_prim(stage, p, basket_material, rgb_fallback=basket_rgb))

    carb.log_info(
        f"Basket material applied; Piper materials are left unchanged from the robot description/USD. "
        f"successful_bind_count={bound_count}"
    )




def create_camera_prim(stage, prim_path, xyz, rpy_deg, focal_length=24.0, clipping_range=(0.01, 1000.0)):
    camera = UsdGeom.Camera.Define(stage, prim_path)
    _set_translate_orient_exact(stage, prim_path, xyz, rpy_deg)
    try:
        camera.CreateFocalLengthAttr(float(focal_length))
    except Exception:
        pass
    try:
        camera.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))
    except Exception:
        pass
    try:
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
    except Exception:
        pass
    return XFormPrim(prim_path)


def create_main_camera(stage):
    return create_camera_prim(
        stage,
        '/World/main_camera',
        args.main_camera_xyz,
        args.main_camera_rpy_deg,
        focal_length=args.main_camera_focal_length,
    )


def create_wrist_camera(stage, robot_prim_path, ee_link_name):
    wrist_camera_path = f"{robot_prim_path}/{ee_link_name}/wrist_camera"
    return create_camera_prim(
        stage,
        wrist_camera_path,
        args.wrist_camera_xyz,
        args.wrist_camera_rpy_deg,
        focal_length=args.wrist_camera_focal_length,
    )


def resolve_place_object_pose(place_marker):
    place_pos, place_quat = place_marker.get_world_pose()
    place_pos = np.array(place_pos, dtype=np.float32)
    place_pos[2] = float(args.place_xyz[2] if args.place_goal_fixed_z is None else args.place_goal_fixed_z)
    place_yaw = yaw_from_quat_wxyz(place_quat)
    return place_pos, place_yaw


def resolve_release_object_pose(place_marker):
    place_pos, place_quat = place_marker.get_world_pose()
    place_pos = np.array(place_pos, dtype=np.float32)
    if not args.use_place_marker_xyz_for_release:
        place_pos[2] = float(args.basket_release_object_center_z)
    place_yaw = yaw_from_quat_wxyz(place_quat)
    return place_pos, place_yaw


def compute_attached_object_place_goals(desired_object_pos, close_rel_offset):
    desired_object_pos = np.array(desired_object_pos, dtype=np.float32)
    close_rel_offset = np.array(close_rel_offset, dtype=np.float32)
    ee_place = desired_object_pos - close_rel_offset
    ee_pre_place = ee_place.copy()
    ee_pre_place[2] += float(args.attached_place_approach_height)

    # Restore the legacy-sized post-release retreat.  The actual return to the
    # retract/home joint configuration is now planned by cuRobo in joint space.
    ee_retreat = ee_pre_place.copy()

    return {
        PickPlaceState.MOVE_PRE_PLACE: ee_pre_place,
        PickPlaceState.MOVE_PLACE: ee_place,
        PickPlaceState.RETREAT: ee_retreat,
    }


def _build_ee_pose_for_attach(tensor_args, ee_prim):
    ee_pos, ee_quat = get_ee_pose(ee_prim)
    return Pose(
        position=tensor_args.to_device(np.asarray(ee_pos, dtype=np.float32)).view(1, 3),
        quaternion=tensor_args.to_device(np.asarray(ee_quat, dtype=np.float32)).view(1, 4),
    )


def best_effort_attach_object(motion_gen, tensor_args, cu_js, ee_prim, object_name_candidates, link_name):
    if args.disable_curobo_attach_detach:
        return False, 'attach/detach disabled by flag'
    attach_fn = getattr(motion_gen, 'attach_objects_to_robot', None)
    if attach_fn is None:
        return False, 'MotionGen.attach_objects_to_robot not available'
    ee_pose = _build_ee_pose_for_attach(tensor_args, ee_prim)
    joint_state = cu_js.unsqueeze(0)
    candidates = []
    for n in object_name_candidates:
        if n not in candidates:
            candidates.append(n)
    last_err = None
    import inspect
    try:
        sig = inspect.signature(attach_fn)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()
    for obj_name in candidates:
        kwargs = {}
        if 'joint_state' in param_names or not param_names:
            kwargs['joint_state'] = joint_state
        if 'object_names' in param_names or not param_names:
            kwargs['object_names'] = [obj_name]
        if 'ee_pose' in param_names or not param_names:
            kwargs['ee_pose'] = ee_pose
        if 'link_name' in param_names or not param_names:
            kwargs['link_name'] = link_name
        if 'scale' in param_names:
            kwargs['scale'] = 1.0
        if 'pitch_scale' in param_names:
            kwargs['pitch_scale'] = 1.0
        if 'merge_meshes' in param_names:
            kwargs['merge_meshes'] = True
        try:
            attach_fn(**kwargs)
            return True, obj_name
        except Exception as exc:
            last_err = exc
    return False, str(last_err) if last_err is not None else 'unknown attach error'


def best_effort_detach_object(motion_gen, object_name_candidates, link_name):
    if args.disable_curobo_attach_detach:
        return False, 'attach/detach disabled by flag'
    detach_fn = getattr(motion_gen, 'detach_object_from_robot', None)
    if detach_fn is None:
        return False, 'MotionGen.detach_object_from_robot not available'
    import inspect
    try:
        sig = inspect.signature(detach_fn)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()
    last_err = None
    candidates = []
    for n in object_name_candidates:
        if n not in candidates:
            candidates.append(n)
    for obj_name in candidates:
        kwargs = {}
        if 'object_names' in param_names or not param_names:
            kwargs['object_names'] = [obj_name]
        if 'link_name' in param_names or not param_names:
            kwargs['link_name'] = link_name
        try:
            detach_fn(**kwargs)
            return True, obj_name
        except Exception as exc:
            last_err = exc
    return False, str(last_err) if last_err is not None else 'unknown detach error'

def main():
    sim_log_dir = None
    sim_log_session_index = None
    task_logger = None
    task_log_index = None

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(xform)
    stage.DefinePrim('/curobo', 'Xform')

    setup_curobo_logger('warn')
    tensor_args = TensorDeviceType()
    usd_help = UsdHelper()

    print(f'[PATH] PROJECT_ROOT={PROJECT_ROOT}')
    print(f'[PATH] robot_config_dir={args.external_robot_configs_path}')
    print(f'[PATH] robot_urdf={args.robot_urdf_path}')
    print(f'[PATH] robot_usd={args.robot_usd_path}')
    print(f'[PATH] robot_meshes={args.external_asset_path}')
    print(f'[PATH] testbed_usd={args.testbed_usd_path}')
    print(f'[PATH] basket_usd={args.basket_usd_path}')
    print(f'[PATH] joint_logs={args.joint_log_dir}')

    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))['robot_cfg']
    if args.robot_usd_path is not None:
        robot_cfg['kinematics']['usd_path'] = str(Path(args.robot_usd_path).expanduser().resolve())
    if args.robot_urdf_path is not None:
        robot_cfg['kinematics']['urdf_path'] = str(Path(args.robot_urdf_path).expanduser().resolve())
    if args.robot_asset_root_path is not None:
        robot_cfg['kinematics']['asset_root_path'] = str(Path(args.robot_asset_root_path).expanduser().resolve())
    if args.external_asset_path is not None:
        robot_cfg['kinematics']['external_asset_path'] = str(Path(args.external_asset_path).expanduser().resolve())
    if args.external_robot_configs_path is not None:
        robot_cfg['kinematics']['external_robot_configs_path'] = str(Path(args.external_robot_configs_path).expanduser().resolve())

    j_names = robot_cfg['kinematics']['cspace']['joint_names']
    default_config = robot_cfg['kinematics']['cspace']['retract_config']
    ee_link_name = robot_cfg['kinematics']['ee_link']

    spawn_center_xy = np.array(args.spawn_region_center_xy if args.spawn_region_center_xy is not None else args.object_xyz[:2], dtype=np.float32)
    spawn_size_xy = np.array(args.spawn_region_size_xy, dtype=np.float32)
    spawn_z = float(args.object_xyz[2] if args.spawn_z is None else args.spawn_z)
    rng = np.random.default_rng(args.random_seed)

    initial_object_pos, initial_object_quat, initial_object_yaw = sample_random_object_pose(rng, spawn_center_xy, spawn_size_xy, spawn_z)
    object_scale = np.array([args.object_long_size, args.object_short_size, args.object_short_size], dtype=np.float32)
    base_goal_quat = np.array(args.orientation, dtype=np.float32)

    pick_object = cuboid.DynamicCuboid(
        '/World/pick_object',
        position=initial_object_pos,
        orientation=initial_object_quat,
        color=np.array([0.2, 0.4, 1.0]),
        size=1.0,
        scale=object_scale,
        mass=args.object_mass,
    )
    basket = create_basket_reference_with_collision(stage)
    place_marker = create_place_marker_xform(stage)
    main_camera = create_main_camera(stage)
    create_workspace_lights(stage)
    create_spawn_region_visuals(spawn_center_xy, spawn_size_xy, spawn_z, args.object_short_size)

    try:
        pick_object.set_solver_position_iteration_count(args.solver_pos_iters)
        pick_object.set_solver_velocity_iteration_count(args.solver_vel_iters)
    except Exception:
        pass

    robot, robot_prim_path, table_prim_path = load_testbed_and_wrap_robot(
        stage, my_world, base_link_name=robot_cfg['kinematics']['base_link'], ee_link_name=ee_link_name
    )
    ee_prim = get_ee_prim(robot_prim_path, ee_link_name=ee_link_name)
    wrist_camera = create_wrist_camera(stage, robot_prim_path, ee_link_name)
    carb.log_info(f"Wrist camera created directly under {robot_prim_path}/{ee_link_name}/wrist_camera with local xyz={args.wrist_camera_xyz}, rpy_deg={args.wrist_camera_rpy_deg} using a single Orient quaternion op")
    apply_basket_material(stage)

    try:
        robot.set_solver_velocity_iteration_count(args.solver_vel_iters)
        robot.set_solver_position_iteration_count(args.solver_pos_iters)
    except Exception:
        pass

    tabletop_proxy_path = create_tabletop_collision_proxy(stage, table_prim_path)
    add_ground_plane(my_world)
    world_cfg = build_initial_table_world_config(stage, tabletop_proxy_path, table_prim_path)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={'obb': 30, 'mesh': 100},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    if not args.reactive:
        print('warming up...')
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print('Curobo is Ready')

    add_extensions(simulation_app, args.headless_mode)

    pose_metric = None
    if args.constrain_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric(offset_position=0.1, tstep_fraction=0.8)

    base_plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
        pose_cost_metric=pose_metric,
    )

    # MOVE_HOME is an exact joint-space target.  Unlike the legacy direct
    # joint stepping, this path is checked by cuRobo against the current world
    # (including the basket after post-detach world synchronization).
    home_plan_config = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=1,
        max_attempts=max(4, max_attempts),
        enable_finetune_trajopt=True,
        time_dilation_factor=float(args.home_time_dilation) if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    try:
        initial_obstacles = usd_help.get_obstacles_from_stage(
            only_paths=['/World'],
            reference_prim_path=robot_prim_path,
            ignore_substring=[robot_prim_path, table_prim_path + '/CAD_Correction', '/World/pick_object', '/World/place_marker', '/World/spawn_region_', '/World/defaultGroundPlane', '/curobo'],
        ).get_collision_check_world()
        motion_gen.update_world(initial_obstacles)
    except Exception as exc:
        carb.log_warn(f'Initial stage obstacle sync failed: {exc}')

    articulation_controller = None
    gripper = None
    joint6_limiter = None
    idx_list = None
    cmd_plan = None
    cmd_idx = 0
    past_cmd = None
    spheres = None

    state = PickPlaceState.SNAPSHOT_OBJECT
    goals = {}
    snapshotted = False
    force_next_plan = False
    last_is_playing = False
    failure_reason = None
    grasp_verified = False
    grasp_close_object_pos = None
    grasp_close_rel_offset = None
    pending_manual_reset = False
    episode_counter = 0
    episode_local_step = 0
    success_counter = 0
    failure_counter = 0
    auto_reset_countdown = None
    spawned_pose = None
    spawn_last_object_pos = None
    spawn_last_object_yaw = None
    spawn_stable_steps = 0
    grasp_joint6_target_rad = None
    locked_grasp_world_yaw_rad = None
    transport_joint6_target_rad = np.deg2rad(args.transport_joint6_deg)
    desired_object_jaw_world_yaw_rad = None
    desired_place_jaw_world_yaw_rad = None
    shape_info = {}
    goal_quats = {}
    startup_initialized = False
    startup_settle_counter = 0
    episode_reset_settle_counter = 0
    curobo_attached_object = False
    curobo_attached_name = None
    home_target_positions = np.array(default_config, dtype=np.float32)
    retry_same_task_after_home = False
    retry_spawn_pose_after_home = None
    retry_attempt_counter = 0

    def save_task_logger(reason):
        nonlocal task_logger, task_log_index
        if task_logger is not None:
            task_logger.close()
            print(f"[LOG] Task joint state CSV saved to {task_logger.csv_path} (task #{task_log_index}, rows={task_logger.rows_written}, reason={reason})")
            task_logger = None
            task_log_index = None

    def discard_task_logger(reason):
        nonlocal task_logger, task_log_index
        if task_logger is not None:
            csv_path = task_logger.csv_path
            rows_written = task_logger.rows_written
            task_logger.close()
            try:
                csv_path.unlink(missing_ok=True)
            except Exception as exc:
                carb.log_warn(f"Failed to remove unsuccessful task log {csv_path}: {exc}")
            print(f"[LOG] Task joint state CSV discarded: {csv_path} (task #{task_log_index}, rows={rows_written}, reason={reason})")
            task_logger = None
            task_log_index = None

    def open_task_logger():
        nonlocal task_logger, task_log_index
        if args.disable_joint_logging:
            return
        if sim_log_dir is None:
            return
        if task_logger is not None:
            discard_task_logger('rotate to next task without finalizing previous log')
        log_path, task_log_index = make_task_log_path(sim_log_dir)
        task_logger = JointStateLogger(log_path, robot.dof_names, args.joint_log_every_n_steps)
        print(f"[LOG] Task joint state logging started: {log_path}")

    def sync_curobo_world(include_pick_object=False):
        ignore_list = [robot_prim_path, table_prim_path + '/CAD_Correction', '/World/place_marker', '/World/spawn_region_', '/World/defaultGroundPlane', '/curobo']
        if not include_pick_object:
            ignore_list.append('/World/pick_object')
        obstacles = usd_help.get_obstacles_from_stage(
            only_paths=['/World'],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        ).get_collision_check_world()
        motion_gen.update_world(obstacles)
        return obstacles

    def begin_new_episode(reason, reset_robot=True, spawn_pose=None, increment_episode=True):
        nonlocal cmd_plan, cmd_idx, past_cmd, idx_list, state, goals, snapshotted
        nonlocal force_next_plan, grasp_verified, grasp_close_object_pos, grasp_close_rel_offset, failure_reason
        nonlocal auto_reset_countdown, spawned_pose, episode_counter, episode_local_step
        nonlocal spawn_last_object_pos, spawn_last_object_yaw, spawn_stable_steps
        nonlocal grasp_joint6_target_rad, locked_grasp_world_yaw_rad, desired_object_jaw_world_yaw_rad, desired_place_jaw_world_yaw_rad, shape_info, goal_quats
        nonlocal curobo_attached_object, curobo_attached_name, retry_same_task_after_home, retry_spawn_pose_after_home, retry_attempt_counter

        if articulation_controller is None or gripper is None:
            return

        arm_idx_list = [robot.get_dof_index(x) for x in j_names]
        if reset_robot:
            reset_episode(robot, gripper, default_config, arm_idx_list, joint6_limiter=joint6_limiter)
        else:
            if joint6_limiter is not None:
                joint6_limiter.disable()
            if gripper is not None:
                gripper.disable_hold()
                gripper.open()
            try:
                robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            except Exception:
                pass

        if spawn_pose is None:
            object_pos, object_quat, object_yaw = sample_random_object_pose(rng, spawn_center_xy, spawn_size_xy, spawn_z)
            spawned_pose = (object_pos.copy(), object_quat.copy(), object_yaw)
            retry_attempt_counter = 0
        else:
            object_pos = np.array(spawn_pose[0], dtype=np.float32)
            object_quat = np.array(spawn_pose[1], dtype=np.float32)
            object_yaw = float(spawn_pose[2])
            spawned_pose = (object_pos.copy(), object_quat.copy(), object_yaw)
            retry_attempt_counter += 1
        respawn_object(pick_object, object_pos, object_quat)

        cmd_plan = None
        cmd_idx = 0
        past_cmd = None
        idx_list = None
        state = PickPlaceState.SNAPSHOT_OBJECT
        goals = {}
        snapshotted = False
        force_next_plan = False
        grasp_verified = False
        grasp_close_object_pos = None
        grasp_close_rel_offset = None
        failure_reason = None
        auto_reset_countdown = None
        if increment_episode:
            episode_counter += 1
        episode_local_step = 0
        spawn_last_object_pos = None
        spawn_last_object_yaw = None
        spawn_stable_steps = 0
        grasp_joint6_target_rad = None
        locked_grasp_world_yaw_rad = None
        desired_object_jaw_world_yaw_rad = None
        desired_place_jaw_world_yaw_rad = None
        shape_info = {}
        goal_quats = {}
        curobo_attached_object = False
        curobo_attached_name = None
        retry_same_task_after_home = False
        retry_spawn_pose_after_home = None

        open_task_logger()
        print(f"[EPISODE] #{episode_counter} start ({reason}) | spawn_object_pos={object_pos}, spawn_yaw_deg={math.degrees(object_yaw):.1f}, reset_robot={reset_robot}, retry_attempt={retry_attempt_counter}")

    while simulation_app.is_running():
        my_world.step(render=True)
        is_playing = my_world.is_playing()

        if not is_playing:
            if last_is_playing:
                pending_manual_reset = True
                discard_task_logger('simulation stopped')
                if sim_log_dir is not None:
                    print(f"[LOG] Simulation joint-log directory closed: {sim_log_dir} (sim #{sim_log_session_index})")
                sim_log_dir = None
                sim_log_session_index = None
                print('[RESET] Stopped. Next Play will restart from a new random object pose.')
            last_is_playing = False
            continue

        just_started = not last_is_playing
        if just_started:
            print('[PLAY] Simulation started.')
            episode_local_step = 0
            if not args.disable_joint_logging:
                sim_log_dir, sim_log_session_index = make_numbered_directory(args.joint_log_dir, 'sim')
                print(f"[LOG] Simulation joint-log directory created: {sim_log_dir} (sim #{sim_log_session_index})")
        else:
            episode_local_step += 1
        last_is_playing = True

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
            gripper = PiperGripperAdapter(robot, hold_effort=args.gripper_hold_effort, hold_kp=args.gripper_hold_kp, hold_kd=args.gripper_hold_kd)
            joint6_limiter = SingleJointRangeLimiter(robot, joint_name='joint6', hold_effort=args.joint6_hold_effort, limit_rad=np.deg2rad(args.joint6_transport_limit_deg))

        arm_idx_list = [robot.get_dof_index(x) for x in j_names]
        step_index = my_world.current_time_step_index

        if not startup_initialized:
            robot._articulation_view.initialize()
            configure_arm_drive_gains(robot, arm_idx_list)
            robot.set_joint_positions(default_config, arm_idx_list)
            robot.set_joint_velocities(np.zeros(len(robot.dof_names), dtype=np.float32))
            set_arm_max_efforts(robot, arm_idx_list, args.startup_arm_effort)
            if gripper is not None:
                gripper.disable_hold()
                gripper.open()
                gripper.configure_hold_drive()
            if joint6_limiter is not None:
                joint6_limiter.disable()
                joint6_limiter.configure_drive()
            startup_initialized = True
            startup_settle_counter = int(args.startup_settle_steps)
            print(f"[INIT] Soft robot initialization applied. Settling for {startup_settle_counter} steps with arm_effort={args.startup_arm_effort:.1f}.")
            continue

        if startup_settle_counter > 0:
            startup_settle_counter -= 1
            if startup_settle_counter == 0:
                set_arm_max_efforts(robot, arm_idx_list, args.nominal_arm_effort)
                print(f"[INIT] Startup settling done. Restored nominal arm_effort={args.nominal_arm_effort:.1f}.")
            continue

        if just_started or pending_manual_reset:
            begin_new_episode('play' if just_started else 'manual reset', reset_robot=True)
            pending_manual_reset = False
            episode_reset_settle_counter = int(args.episode_reset_settle_steps)
            if episode_reset_settle_counter > 0:
                set_arm_max_efforts(robot, arm_idx_list, args.startup_arm_effort)
                print(f"[RESET] Episode reset settling for {episode_reset_settle_counter} steps with arm_effort={args.startup_arm_effort:.1f}.")
            else:
                set_arm_max_efforts(robot, arm_idx_list, args.nominal_arm_effort)
            continue

        if episode_reset_settle_counter > 0:
            episode_reset_settle_counter -= 1
            if joint6_limiter is not None:
                joint6_limiter.disable()
            if gripper is not None and gripper.hold_enabled:
                gripper.disable_hold()
            if episode_reset_settle_counter == 0:
                set_arm_max_efforts(robot, arm_idx_list, args.nominal_arm_effort)
                print(f"[RESET] Episode reset settling done. Restored nominal arm_effort={args.nominal_arm_effort:.1f}.")
            continue

        if step_index == 50 or step_index % 1000 == 0:
            try:
                sync_curobo_world(include_pick_object=False)
                carb.log_info('Synced CuRobo world from stage.')
            except Exception as exc:
                carb.log_warn(f'Periodic CuRobo world sync failed: {exc}')

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        if np.any(np.isnan(sim_js.positions)):
            log_error('Isaac Sim returned NaN joint values.')
            continue

        if task_logger is not None:
            task_logger.maybe_log(step_index, my_world.current_time, state, sim_js)

        if gripper is not None and gripper.hold_enabled and not args.disable_continuous_gripper_hold:
            gripper.hold_step()
        if joint6_limiter is not None and joint6_limiter.enabled:
            joint6_limiter.hold_step()

        sim_js_names = robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for si, s in enumerate(sph_list[0]):
                    try:
                        pos = np.ravel(s[:3].cpu().numpy())
                        radius = float(s[3].cpu().numpy())
                    except Exception:
                        pos = np.ravel(s.position)
                        radius = float(s.radius)
                    sp = sphere.VisualSphere(prim_path=f'/curobo/robot_sphere_{si}', position=pos, radius=radius, color=np.array([0.2, 0.8, 0.2]))
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    try:
                        pos = np.ravel(s[:3].cpu().numpy())
                        radius = float(s[3].cpu().numpy())
                    except Exception:
                        pos = np.ravel(s.position)
                        radius = float(s.radius)
                    if not np.isnan(pos[0]):
                        spheres[si].set_world_pose(position=pos)
                        spheres[si].set_radius(radius)

        arm_static, arm_vel = is_arm_static(robot, sim_js, j_names, args.arm_static_threshold)
        robot_static = arm_static or args.reactive

        if state == PickPlaceState.SNAPSHOT_OBJECT and robot_static and not snapshotted and episode_local_step >= args.snapshot_wait_steps:
            if episode_local_step < args.snapshot_wait_steps + args.spawn_force_zero_steps:
                pick_object.set_linear_velocity(np.zeros(3, dtype=np.float32))
                pick_object.set_angular_velocity(np.zeros(3, dtype=np.float32))

            obj_lin = np.linalg.norm(np.array(pick_object.get_linear_velocity(), dtype=np.float32))
            obj_ang = np.linalg.norm(np.array(pick_object.get_angular_velocity(), dtype=np.float32))
            object_pos, object_quat_now = pick_object.get_world_pose()
            object_pos = np.array(object_pos, dtype=np.float32)
            object_yaw_now = yaw_from_quat_wxyz(object_quat_now)

            if spawn_last_object_pos is None:
                spawn_last_object_pos = object_pos.copy()
                spawn_last_object_yaw = object_yaw_now

            pos_delta = float(np.linalg.norm(object_pos - spawn_last_object_pos))
            yaw_delta = abs(wrap_to_pi(object_yaw_now - spawn_last_object_yaw))
            vel_ok = obj_lin < args.spawn_settle_lin_vel_thresh and obj_ang < args.spawn_settle_ang_vel_thresh
            pose_ok = pos_delta < args.spawn_settle_pos_eps and yaw_delta < math.radians(args.spawn_settle_yaw_eps_deg)

            if vel_ok or pose_ok:
                spawn_stable_steps += 1
            else:
                spawn_stable_steps = 0

            spawn_last_object_pos = object_pos.copy()
            spawn_last_object_yaw = object_yaw_now

            timeout_reached = episode_local_step >= args.snapshot_wait_steps + args.spawn_settle_timeout_steps
            if spawn_stable_steps >= args.spawn_settle_stable_steps or timeout_reached:
                place_pos, place_yaw_now = resolve_place_object_pose(place_marker)
                plan_data = compute_top_grasp_plan(object_pos, object_yaw_now, place_pos, place_yaw_now)
                goals = plan_data['goals']
                desired_object_jaw_world_yaw_rad = plan_data['desired_object_jaw_world_yaw_rad']
                desired_place_jaw_world_yaw_rad = plan_data['desired_place_jaw_world_yaw_rad']
                transport_joint6_target_rad = plan_data['transport_joint6_target_rad']
                shape_info = plan_data.get('shape_info', {})
                goal_quats = {PickPlaceState.MOVE_PRE_PICK: base_goal_quat.copy()}
                snapshotted = True
                grasp_verified = False
                locked_grasp_world_yaw_rad = None
                state = PickPlaceState.MOVE_PRE_PICK
                force_next_plan = True
                if joint6_limiter is not None:
                    joint6_limiter.enable(transport_joint6_target_rad, np.deg2rad(args.joint6_transport_limit_deg))
                reason = 'timeout' if timeout_reached and spawn_stable_steps < args.spawn_settle_stable_steps else 'stable'
                print(f"[SNAPSHOT] object accepted ({reason}) | pos={object_pos}, yaw_deg={math.degrees(object_yaw_now):.2f}, lin_vel={obj_lin:.4f}, ang_vel={obj_ang:.4f}")
                print(f"[SNAPSHOT] desired jaw-closing yaw(world, canonical)={math.degrees(desired_object_jaw_world_yaw_rad):.2f} deg | grasp_mode={args.grasp_direction_mode} | lock_grasp_branch_during_prepick={args.lock_grasp_branch_during_prepick}")
                if shape_info:
                    print(f"[SNAPSHOT] top_face_z={shape_info.get('object_top_face_z', float('nan')):.4f}, pick_target_z={shape_info.get('pick_target_z', float('nan')):.4f}, place_target_z={shape_info.get('place_target_z', float('nan')):.4f}")
                print(f"[STATE] -> {state}")
                continue
            elif step_index % 20 == 0:
                print(f"[WAIT] object settling in spawn region | lin_vel={obj_lin:.4f}, ang_vel={obj_ang:.4f}, pos_delta={pos_delta:.5f}, yaw_delta_deg={math.degrees(yaw_delta):.2f}, stable_steps={spawn_stable_steps}/{args.spawn_settle_stable_steps}")

        if state in {PickPlaceState.MOVE_PRE_PLACE, PickPlaceState.MOVE_PLACE} and grasp_verified and grasp_close_rel_offset is not None:
            carried, object_pos_now, ee_pos_now, carry_error = is_object_still_carried(pick_object, ee_prim, grasp_close_rel_offset)
            if not carried:
                failure_reason = f"Object appears dropped during transport. carry_error={carry_error:.4f}, object_pos={object_pos_now}, ee_pos={ee_pos_now}"
                state = PickPlaceState.FAILED
                print(f"[FAIL] {failure_reason}")
                continue

        plan_states = {
            PickPlaceState.MOVE_PRE_PICK,
            PickPlaceState.MOVE_PICK,
            PickPlaceState.CLOSE_AND_LIFT,
            PickPlaceState.MOVE_PRE_PLACE,
            PickPlaceState.MOVE_PLACE,
            PickPlaceState.RETREAT,
            PickPlaceState.MOVE_HOME,
        }
        can_start_next_plan = robot_static or force_next_plan or args.reactive

        if state in plan_states and cmd_plan is None and can_start_next_plan:
            if state == PickPlaceState.MOVE_PRE_PICK and joint6_limiter is not None:
                joint6_limiter.disable()
            elif state in {PickPlaceState.MOVE_PRE_PLACE, PickPlaceState.MOVE_PLACE, PickPlaceState.RETREAT} and joint6_limiter is not None:
                joint6_limiter.enable(transport_joint6_target_rad, np.deg2rad(args.joint6_transport_limit_deg))
            elif state in {PickPlaceState.MOVE_PICK, PickPlaceState.CLOSE_AND_LIFT} and joint6_limiter is not None and grasp_joint6_target_rad is not None:
                joint6_limiter.enable(grasp_joint6_target_rad, np.deg2rad(args.joint6_grasp_limit_deg))

            if state == PickPlaceState.CLOSE_AND_LIFT and gripper is not None and not gripper.hold_enabled:
                print('[GRIPPER] close + lift start')
                gripper.close()
                gripper.enable_hold(gripper.closed_positions)
                object_pos_now, ee_pos_now = get_object_and_ee_pose(pick_object, ee_prim)
                grasp_close_object_pos = object_pos_now.copy()
                grasp_close_rel_offset = object_pos_now - ee_pos_now
                print('[GRASP] object pose at close+lift start:', grasp_close_object_pos)
                print('[GRASP] object center relative to ee:', grasp_close_rel_offset)

            if state == PickPlaceState.MOVE_HOME:
                if gripper is not None:
                    gripper.disable_hold()
                    gripper.open()
                if joint6_limiter is not None:
                    joint6_limiter.disable()

                cmd_plan, idx_list = build_joint_plan(
                    motion_gen,
                    tensor_args,
                    home_plan_config,
                    robot,
                    sim_js,
                    home_target_positions,
                    j_names,
                )
                goal_position = home_target_positions
                if cmd_plan is not None:
                    print(
                        f"[PLAN] state={state}, cuRobo joint-space home target="
                        f"{np.rad2deg(home_target_positions)} deg"
                    )
            else:
                goal_position = goals[state]
                goal_quat = goal_quats.get(state, base_goal_quat)
                selected_plan_config = base_plan_config
                if state in {PickPlaceState.MOVE_PRE_PLACE, PickPlaceState.MOVE_PLACE, PickPlaceState.RETREAT}:
                    selected_plan_config = MotionGenPlanConfig(
                        enable_graph=False,
                        enable_graph_attempt=2,
                        max_attempts=max_attempts,
                        enable_finetune_trajopt=enable_finetune_trajopt,
                        time_dilation_factor=args.transport_time_dilation if not args.reactive else 1.0,
                        pose_cost_metric=pose_metric,
                    )
                cmd_plan, idx_list = build_plan(
                    motion_gen,
                    tensor_args,
                    selected_plan_config,
                    robot,
                    sim_js,
                    goal_position,
                    goal_quat,
                )

            if cmd_plan is not None:
                if joint6_limiter is not None and joint6_limiter.enabled:
                    for row_idx in range(len(cmd_plan.position)):
                        try:
                            row_np = cmd_plan.position[row_idx].cpu().numpy()
                            row_np = joint6_limiter.clamp_joint_array_in_place(row_np, list(cmd_plan.joint_names))
                            cmd_plan.position[row_idx] = torch.as_tensor(row_np, device=cmd_plan.position.device)
                        except Exception:
                            pass
                cmd_idx = 0
                force_next_plan = False
                if state != PickPlaceState.MOVE_HOME:
                    print(f"[PLAN] state={state}, goal={goal_position}")
            else:
                failure_reason = f"Motion planning failed at state={state}, goal={goal_position}"
                state = PickPlaceState.FAILED
                print(f"[FAIL] {failure_reason}")
                continue

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            action = ArticulationAction(cmd_state.position.cpu().numpy(), cmd_state.velocity.cpu().numpy() * 0.0, joint_indices=idx_list)
            if joint6_limiter is not None and joint6_limiter.enabled:
                action = joint6_limiter.apply_to_arm_action(action, idx_list)
            articulation_controller.apply_action(action)

            if state == PickPlaceState.MOVE_PRE_PICK and desired_object_jaw_world_yaw_rad is not None:
                if joint6_limiter is not None:
                    joint6_limiter.disable()
                if args.lock_grasp_branch_during_prepick and locked_grasp_world_yaw_rad is None:
                    current_jaw_world_yaw, _, _ = get_current_jaw_world_yaw(ee_prim)
                    locked_grasp_world_yaw_rad = choose_grasp_world_yaw_branch(
                        desired_object_jaw_world_yaw_rad,
                        current_jaw_world_yaw,
                    )
                    print(
                        f"[JOINT6] locked grasp branch for MOVE_PRE_PICK | "
                        f"jaw_world_target={np.rad2deg(locked_grasp_world_yaw_rad):.2f} deg"
                    )
                if args.lock_grasp_branch_during_prepick and locked_grasp_world_yaw_rad is not None:
                    solved = solve_joint6_target_for_world_yaw(robot, ee_prim, locked_grasp_world_yaw_rad)
                else:
                    solved = solve_dynamic_joint6_target(robot, ee_prim, desired_object_jaw_world_yaw_rad)
                if solved is not None:
                    grasp_joint6_target_rad = solved['target_joint6_rad']
                    _, current_rad, cmd_rad = drive_joint_toward(robot, 'joint6', grasp_joint6_target_rad, np.deg2rad(args.joint6_align_step_deg))
                    if step_index % 20 == 0:
                        print(
                            f"[JOINT6] pre-pick align | current_joint6={np.rad2deg(current_rad):.2f} deg, "
                            f"cmd_joint6={np.rad2deg(cmd_rad):.2f} deg, target_joint6={np.rad2deg(grasp_joint6_target_rad):.2f} deg, "
                            f"jaw_world_current={np.rad2deg(solved['current_jaw_world_yaw_rad']):.2f} deg, "
                            f"jaw_world_target={np.rad2deg(solved['desired_target_world_yaw_rad']):.2f} deg"
                        )

            if gripper is not None and gripper.hold_enabled and not args.disable_continuous_gripper_hold:
                gripper.hold_step()
            if joint6_limiter is not None and joint6_limiter.enabled:
                joint6_limiter.hold_step()
            cmd_idx += 1
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                if state == PickPlaceState.MOVE_PRE_PICK:
                    _, aligned_ee_quat = get_ee_pose(ee_prim)
                    goal_quats[PickPlaceState.MOVE_PICK] = aligned_ee_quat.copy()
                    goal_quats[PickPlaceState.CLOSE_AND_LIFT] = aligned_ee_quat.copy()
                    if joint6_limiter is not None and grasp_joint6_target_rad is not None:
                        joint6_limiter.enable(grasp_joint6_target_rad, np.deg2rad(args.joint6_grasp_limit_deg))
                    state = PickPlaceState.MOVE_PICK
                    force_next_plan = True
                elif state == PickPlaceState.MOVE_PICK:
                    state = PickPlaceState.CLOSE_AND_LIFT
                    force_next_plan = True
                elif state == PickPlaceState.CLOSE_AND_LIFT:
                    settled, settle_vel = settle_after_gripper(my_world, robot, j_names, args.settle_threshold, args.post_gripper_settle_steps, gripper=gripper, joint6_limiter=joint6_limiter)
                    if settle_vel is not None:
                        print(f"[SETTLE] after close+lift, settled={settled}, arm_max_vel={np.max(np.abs(settle_vel)):.4f}")
                    state = PickPlaceState.VERIFY_GRASP
                    force_next_plan = False
                elif state == PickPlaceState.MOVE_PRE_PLACE:
                    state = PickPlaceState.MOVE_PLACE
                    force_next_plan = True
                elif state == PickPlaceState.MOVE_PLACE:
                    state = PickPlaceState.OPEN_GRIPPER
                    force_next_plan = False
                elif state == PickPlaceState.RETREAT:
                    state = PickPlaceState.MOVE_HOME
                    force_next_plan = True
                elif state == PickPlaceState.MOVE_HOME:
                    try:
                        robot.set_joint_velocities(
                            np.zeros(len(robot.dof_names), dtype=np.float32)
                        )
                    except Exception:
                        pass

                    if retry_same_task_after_home and retry_spawn_pose_after_home is not None:
                        retry_pose = retry_spawn_pose_after_home
                        retry_same_task_after_home = False
                        retry_spawn_pose_after_home = None
                        print(
                            '[HOME] cuRobo return to default configuration done. '
                            'Retrying the same task pose.'
                        )
                        begin_new_episode(
                            'failure retry same pose',
                            reset_robot=False,
                            spawn_pose=retry_pose,
                            increment_episode=False,
                        )
                        # begin_new_episode sets the next state itself.
                    else:
                        state = PickPlaceState.DONE
                        force_next_plan = False
                        print('[HOME] cuRobo return to default configuration done.')
                print(f"[STATE] -> {state}")
            continue

        if state == PickPlaceState.VERIFY_GRASP:
            if grasp_close_object_pos is None or grasp_close_rel_offset is None:
                failure_reason = 'Grasp verification requested without a recorded close pose.'
                state = PickPlaceState.FAILED
                print(f"[FAIL] {failure_reason}")
                continue
            success, info = evaluate_grasp_success(pick_object, ee_prim, grasp_close_object_pos, grasp_close_rel_offset)
            print('[VERIFY] lift_amount=', f"{info['lift_amount']:.4f}", 'carry_error=', f"{info['carry_error']:.4f}", 'object_pos=', info['object_pos'], 'ee_pos=', info['ee_pos'])
            if success:
                grasp_verified = True
                if gripper is not None:
                    gripper.enable_hold(gripper.closed_positions)
                try:
                    desired_place_object_pos, desired_place_yaw = resolve_release_object_pose(place_marker)
                except Exception:
                    desired_place_object_pos = np.array(args.place_xyz, dtype=np.float32)
                    desired_place_object_pos[2] = float(args.basket_release_object_center_z)
                    desired_place_yaw = np.deg2rad(args.place_object_yaw_deg)
                attached_goals = compute_attached_object_place_goals(desired_place_object_pos, grasp_close_rel_offset)
                goals[PickPlaceState.MOVE_PRE_PLACE] = attached_goals[PickPlaceState.MOVE_PRE_PLACE]
                goals[PickPlaceState.MOVE_PLACE] = attached_goals[PickPlaceState.MOVE_PLACE]
                goals[PickPlaceState.RETREAT] = attached_goals[PickPlaceState.RETREAT]
                desired_place_jaw_world_yaw_rad = canonicalize_mod_pi(float(desired_place_yaw) + np.deg2rad((0.0 if args.grasp_direction_mode == 'top_grasp_short_axis' else 90.0) + args.grasp_yaw_trim_deg))
                try:
                    sync_curobo_world(include_pick_object=True)
                except Exception as exc:
                    carb.log_warn(f'Pre-attach CuRobo world sync failed: {exc}')
                attached_ok, attached_info = best_effort_attach_object(
                    motion_gen,
                    tensor_args,
                    cu_js,
                    ee_prim,
                    ['/World/pick_object', 'pick_object'],
                    ee_link_name,
                )
                curobo_attached_object = bool(attached_ok)
                curobo_attached_name = attached_info if attached_ok else None
                if attached_ok:
                    print(f"[CUROBO] Attached object to robot for transport collision checking: {attached_info}")
                else:
                    print(f"[CUROBO] Attach skipped/failed: {attached_info}")
                state = PickPlaceState.REORIENT_FOR_TRANSPORT
                force_next_plan = False
                print('[VERIFY] Physical top grasp accepted.')
                print(f"[GOAL] basket-release pre_place ee={goals[PickPlaceState.MOVE_PRE_PLACE]}, release ee={goals[PickPlaceState.MOVE_PLACE]}, retreat ee={goals[PickPlaceState.RETREAT]}, attached_place_approach_height={args.attached_place_approach_height:.3f}, home_mode=curobo_plan_single_js, release_object_center_z={desired_place_object_pos[2]:.3f}, use_place_marker_xyz_for_release={args.use_place_marker_xyz_for_release}")
                print(f"[STATE] -> {state}")
                continue
            failure_reason = f"Physical grasp verification failed: lift_amount={info['lift_amount']:.4f}, carry_error={info['carry_error']:.4f}"
            state = PickPlaceState.FAILED
            print(f"[FAIL] {failure_reason}")
            continue

        if state == PickPlaceState.REORIENT_FOR_TRANSPORT:
            if joint6_limiter is not None:
                joint6_limiter.disable()
            done_align, current_rad, cmd_rad = drive_joint_toward(robot, 'joint6', transport_joint6_target_rad, np.deg2rad(args.joint6_align_step_deg))
            if done_align:
                if joint6_limiter is not None:
                    joint6_limiter.enable(transport_joint6_target_rad, np.deg2rad(args.joint6_transport_limit_deg))
                _, transport_ee_quat = get_ee_pose(ee_prim)
                goal_quats[PickPlaceState.MOVE_PRE_PLACE] = transport_ee_quat.copy()
                goal_quats[PickPlaceState.MOVE_PLACE] = transport_ee_quat.copy()
                goal_quats[PickPlaceState.RETREAT] = transport_ee_quat.copy()
                settled, settle_vel = settle_after_gripper(my_world, robot, j_names, args.settle_threshold, args.post_gripper_settle_steps, gripper=gripper, joint6_limiter=joint6_limiter)
                if settle_vel is not None:
                    print(f"[SETTLE] after transport reorient, settled={settled}, arm_max_vel={np.max(np.abs(settle_vel)):.4f}")
                state = PickPlaceState.MOVE_PRE_PLACE
                force_next_plan = True
                print(f"[JOINT6] transport reorientation done. current={np.rad2deg(current_rad):.2f} deg, target={np.rad2deg(transport_joint6_target_rad):.2f} deg")
                print(f"[STATE] -> {state}")
            elif step_index % 10 == 0:
                print(f"[JOINT6] aligning for transport | current={np.rad2deg(current_rad):.2f} deg, cmd={np.rad2deg(cmd_rad):.2f} deg, target={np.rad2deg(transport_joint6_target_rad):.2f} deg")
            continue

        if state == PickPlaceState.OPEN_GRIPPER and gripper is not None:
            print('[GRIPPER] open')
            gripper.disable_hold()
            gripper.open()
            for _ in range(args.wait_steps_after_gripper):
                if joint6_limiter is not None and joint6_limiter.enabled:
                    joint6_limiter.hold_step()
                my_world.step(render=True)
            settled, settle_vel = settle_after_gripper(my_world, robot, j_names, args.settle_threshold, args.post_gripper_settle_steps, gripper=gripper, joint6_limiter=joint6_limiter)
            if settle_vel is not None:
                print(f"[SETTLE] after open, settled={settled}, arm_max_vel={np.max(np.abs(settle_vel)):.4f}")
            if curobo_attached_object:
                detached_ok, detached_info = best_effort_detach_object(motion_gen, [curobo_attached_name, '/World/pick_object', 'pick_object'], ee_link_name)
                if detached_ok:
                    print(f"[CUROBO] Detached object from robot: {detached_info}")
                else:
                    print(f"[CUROBO] Detach skipped/failed: {detached_info}")
                curobo_attached_object = False
                curobo_attached_name = None
                try:
                    sync_curobo_world(include_pick_object=True)
                except Exception as exc:
                    carb.log_warn(f'Post-detach CuRobo world sync failed: {exc}')
            state = PickPlaceState.RETREAT
            force_next_plan = True
            print(f"[STATE] -> {state}")
            continue

        if state == PickPlaceState.DONE:
            place_success, object_pos_now, place_pos_now, xy_error, z_error = report_final_place_quality(pick_object, place_marker)
            if auto_reset_countdown is None:
                save_task_logger('task success')
                success_counter += 1
                print(f"[DONE] episode #{episode_counter} finished.")
                print('[RESULT] object world position:', object_pos_now)
                print('[RESULT] place marker position:', place_pos_now)
                print(f"[RESULT] xy_error={xy_error:.4f}, z_error={z_error:.4f}")
                if place_success:
                    print('[RESULT] Placement is within the configured XY tolerance.')
                else:
                    print('[RESULT] Placement finished, but final place error exceeds the configured tolerance.')
                print(f"[SUMMARY] successes={success_counter}, failures={failure_counter}, max_episodes={'inf' if args.max_episodes == 0 else args.max_episodes}")
                if args.max_episodes > 0 and success_counter >= args.max_episodes:
                    auto_reset_countdown = -1
                    print('[STOP] Reached max_episodes. Leaving the robot at the default home configuration.')
                else:
                    auto_reset_countdown = args.episode_pause_steps
            elif auto_reset_countdown > 0:
                auto_reset_countdown -= 1
            elif auto_reset_countdown == 0:
                begin_new_episode('success repeat', reset_robot=False)
            continue

        if state == PickPlaceState.FAILED:
            if gripper is not None:
                gripper.disable_hold()
                gripper.open()
            if joint6_limiter is not None:
                joint6_limiter.disable()
            if auto_reset_countdown is None:
                discard_task_logger('task failure')
                failure_counter += 1
                if curobo_attached_object:
                    detached_ok, detached_info = best_effort_detach_object(motion_gen, [curobo_attached_name, '/World/pick_object', 'pick_object'], ee_link_name)
                    if detached_ok:
                        print(f"[CUROBO] Detached object from robot after failure: {detached_info}")
                    else:
                        print(f"[CUROBO] Failure detach skipped/failed: {detached_info}")
                    curobo_attached_object = False
                    curobo_attached_name = None
                    try:
                        sync_curobo_world(include_pick_object=True)
                    except Exception as exc:
                        carb.log_warn(f'Failure-path CuRobo world sync failed: {exc}')
                auto_reset_countdown = 0
                retry_same_task_after_home = True
                retry_spawn_pose_after_home = spawned_pose
                print(f"[FAILED] episode #{episode_counter} attempt failed.")
                print('[RESULT]', failure_reason)
                print(f"[SUMMARY] successes={success_counter}, failures={failure_counter}")
                print('[RETRY] Discarded failed task log. Returning home, then respawning the cube at the same task pose for retry.')
            elif auto_reset_countdown > 0:
                auto_reset_countdown -= 1
            elif auto_reset_countdown == 0:
                state = PickPlaceState.MOVE_HOME
                force_next_plan = True
                auto_reset_countdown = -1
                print(f"[STATE] -> {state}")
            continue


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
