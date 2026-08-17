#!/usr/bin/env python3
"""
Load configs/camera/main_camera.yaml and place an Isaac Sim Camera prim at the
calibrated Piper-World pose.

Purpose
-------
This is a VISUAL VERIFICATION utility. It does not modify the calibration YAML.

It:
  1. Opens the Piper testbed USD (auto-detected or supplied with --stage-usd).
  2. Loads calibration.T_world_isaac_camera_usd from main_camera.yaml.
  3. Creates/updates /World/main_camera.
  4. Applies the calibrated 640x480 intrinsics.
  5. Switches the active viewport to the calibrated camera when possible.
  6. Prints the requested pose and the pose read back from Isaac Sim.

Important distortion note
-------------------------
For the current RealSense inverse_brown_conrady calibration, the YAML contains
the raw RealSense inverse-distortion coefficients for provenance, but they are
NOT OpenCV forward Brown-Conrady coefficients. For this placement-check script
the simulated camera therefore uses the calibrated K matrix as an UNDISTORTED
OpenCV pinhole camera. This is sufficient to verify camera pose and FOV/framing.

Run, for example:
    cd ~/icros_journal
    omni_python src/camera/verify_main_camera_from_yaml.py

If auto-detection cannot find the testbed:
    omni_python src/camera/verify_main_camera_from_yaml.py \
        --stage-usd /absolute/path/to/piper_testbed.usd

Optional:
    --camera-yaml /absolute/path/to/main_camera.yaml
    --camera-prim /World/main_camera
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# CLI and project paths
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify calibrated main-camera placement in Isaac Sim."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Auto-detected from cwd/script when omitted.",
    )
    parser.add_argument(
        "--camera-yaml",
        default=None,
        help="Default: <project>/configs/camera/main_camera.yaml",
    )
    parser.add_argument(
        "--stage-usd",
        default=None,
        help="Piper testbed USD. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--camera-prim",
        default="/World/main_camera",
        help="USD Camera prim path.",
    )
    parser.add_argument(
        "--near-clip",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--far-clip",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
    )
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_args()


def candidate_roots() -> list[Path]:
    candidates: list[Path] = []

    if ARGS.project_root:
        candidates.append(Path(ARGS.project_root).expanduser().resolve())

    try:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])
    except Exception:
        pass

    try:
        here = Path(__file__).resolve().parent
        candidates.extend([here, *here.parents])
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def find_project_root() -> Path:
    if ARGS.project_root:
        root = Path(ARGS.project_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--project-root does not exist: {root}")
        return root

    for p in candidate_roots():
        if (p / "configs" / "camera").is_dir():
            return p

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()

CAMERA_YAML = (
    Path(ARGS.camera_yaml).expanduser().resolve()
    if ARGS.camera_yaml
    else (PROJECT_ROOT / "configs" / "camera" / "main_camera.yaml").resolve()
)


def find_stage_usd() -> Path:
    if ARGS.stage_usd:
        p = Path(ARGS.stage_usd).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--stage-usd does not exist: {p}")
        return p

    candidates = [
        PROJECT_ROOT / "assets" / "piper_testbed" / "piper_testbed.usd",
        PROJECT_ROOT / "assets" / "piper_testbed" / "piper_testbed.usda",
        PROJECT_ROOT / "assets" / "piper_testbed.usd",
        PROJECT_ROOT / "assets" / "robot_testbed" / "piper_testbed.usd",
    ]

    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        "Could not auto-detect the Piper testbed USD.\n"
        "Pass it explicitly, for example:\n"
        "  --stage-usd /absolute/path/to/piper_testbed.usd\n"
        "Checked:\n  " + "\n  ".join(str(p) for p in candidates)
    )


STAGE_USD = find_stage_usd()


# -----------------------------------------------------------------------------
# Load and validate YAML before launching Isaac Sim
# -----------------------------------------------------------------------------
def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected mapping '{key}' in {CAMERA_YAML}")
    return value


def rotation_is_valid(R: np.ndarray, atol: float = 1e-4) -> bool:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return bool(
        np.all(np.isfinite(R))
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to scalar-first [w,x,y,z]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    # Project tiny numerical drift back to SO(3).
    u, _s, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vh

    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


if not CAMERA_YAML.is_file():
    raise FileNotFoundError(f"Camera YAML not found: {CAMERA_YAML}")

with CAMERA_YAML.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

if not isinstance(CFG, dict):
    raise ValueError(f"YAML root must be a mapping: {CAMERA_YAML}")

camera_cfg = require_mapping(CFG, "camera")
intr_cfg = require_mapping(CFG, "intrinsics")
calib_cfg = require_mapping(CFG, "calibration")

T_world_camera = np.asarray(
    calib_cfg["T_world_isaac_camera_usd"], dtype=np.float64
).reshape(4, 4)

if not np.allclose(T_world_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
    raise ValueError("Invalid homogeneous transform bottom row.")
if not rotation_is_valid(T_world_camera[:3, :3]):
    raise ValueError("T_world_isaac_camera_usd contains an invalid rotation.")

POSITION = T_world_camera[:3, 3].copy()
QUAT_WXYZ = rotation_matrix_to_quaternion_wxyz(T_world_camera[:3, :3])

WIDTH = int(camera_cfg["width"])
HEIGHT = int(camera_cfg["height"])
FPS = int(camera_cfg["fps"])

FX = float(intr_cfg["fx"])
FY = float(intr_cfg["fy"])
CX = float(intr_cfg["cx"])
CY = float(intr_cfg["cy"])

DIST_MODEL = str(intr_cfg.get("distortion_model_realsense", "unknown"))
PNP_DISTORTION_HANDLING = str(
    intr_cfg.get("pnp_distortion_handling", "unknown")
)

opencv_coeffs = np.asarray(
    intr_cfg.get(
        "distortion_coefficients_opencv_k1_k2_p1_p2_k3",
        [0.0] * 5,
    ),
    dtype=np.float64,
).reshape(-1)

# OpenCV pinhole in Isaac accepts up to:
# [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
PINHOLE_COEFFS = np.zeros(12, dtype=np.float64)
PINHOLE_COEFFS[: min(12, len(opencv_coeffs))] = opencv_coeffs[:12]

# Inverse Brown must not be treated as forward OpenCV Brown distortion.
USE_IDEAL_PINHOLE = (
    DIST_MODEL == "inverse_brown_conrady"
    or PNP_DISTORTION_HANDLING == "realsense_deproject_to_pinhole"
)
if USE_IDEAL_PINHOLE:
    PINHOLE_COEFFS[:] = 0.0


print("\n" + "=" * 72)
print("Calibrated main-camera placement verification")
print(f"Project root : {PROJECT_ROOT}")
print(f"Camera YAML  : {CAMERA_YAML}")
print(f"Stage USD    : {STAGE_USD}")
print(f"Camera prim  : {ARGS.camera_prim}")
print(f"Resolution   : {WIDTH}x{HEIGHT} @ {FPS} FPS")
print(f"K            : fx={FX:.4f}, fy={FY:.4f}, cx={CX:.4f}, cy={CY:.4f}")
print(f"RS distortion: {DIST_MODEL}")
print(f"PnP handling : {PNP_DISTORTION_HANDLING}")
print(
    "YAML USD xyz : "
    f"[{POSITION[0]:.6f}, {POSITION[1]:.6f}, {POSITION[2]:.6f}] m"
)
print(
    "YAML USD quat: "
    f"[{QUAT_WXYZ[0]:.7f}, {QUAT_WXYZ[1]:.7f}, "
    f"{QUAT_WXYZ[2]:.7f}, {QUAT_WXYZ[3]:.7f}] wxyz"
)
if USE_IDEAL_PINHOLE:
    print(
        "Lens model   : ideal OpenCV pinhole using calibrated K "
        "(raw inverse-Brown distortion is NOT applied)"
    )
print("=" * 72 + "\n")


# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------
try:
    from isaacsim import SimulationApp
except ImportError:
    from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": int(ARGS.window_width),
        "height": int(ARGS.window_height),
    }
)


# Imports that require a running Kit app.
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from isaacsim.core.utils.stage import is_stage_loading, open_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402


def wait_for_stage(max_updates: int = 600) -> None:
    for _ in range(max_updates):
        simulation_app.update()
        if not is_stage_loading():
            # A few extra updates help renderer/resource initialization.
            for _j in range(8):
                simulation_app.update()
            return
    raise TimeoutError("USD stage did not finish loading.")


def switch_active_viewport_to_camera(
    camera_prim_path: str,
    resolution: tuple[int, int],
) -> bool:
    """
    Prefer Isaac's viewport manager. Fall back to Kit viewport API variants.
    This helper intentionally does not fail the entire test if viewport switching
    API differs in a local Isaac build.
    """
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera(camera_prim_path)
        ViewportManager.set_resolution(resolution)
        print(
            f"[OK] Active viewport -> {camera_prim_path}, "
            f"resolution={resolution[0]}x{resolution[1]}"
        )
        return True
    except Exception as exc:
        print(f"[WARN] ViewportManager path failed: {exc}")

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport.")

        # Kit API variants differ by release.
        if hasattr(viewport, "camera_path"):
            viewport.camera_path = Sdf.Path(camera_prim_path)
        elif hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(camera_prim_path)
        else:
            raise RuntimeError("No supported camera-switch method on viewport.")

        if hasattr(viewport, "resolution"):
            viewport.resolution = resolution
        elif hasattr(viewport, "set_texture_resolution"):
            viewport.set_texture_resolution(resolution)

        print(
            f"[OK] Active viewport -> {camera_prim_path} "
            "(Kit viewport fallback)"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Automatic viewport switch failed: {exc}")
        print(
            "[INFO] Camera prim was still created. In Isaac Sim, use the viewport "
            "camera menu and select the calibrated camera manually."
        )
        return False


# -----------------------------------------------------------------------------
# Open scene and place camera
# -----------------------------------------------------------------------------
ok = open_stage(str(STAGE_USD))
if not ok:
    raise RuntimeError(f"Isaac Sim could not open stage: {STAGE_USD}")

wait_for_stage()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("USD stage is unavailable after open_stage().")

existing = stage.GetPrimAtPath(ARGS.camera_prim)
if existing and existing.IsValid():
    if not existing.IsA(UsdGeom.Camera):
        raise RuntimeError(
            f"{ARGS.camera_prim} already exists but is not a USD Camera "
            f"(type={existing.GetTypeName()}). Choose another --camera-prim."
        )
    print(f"[INFO] Reusing existing Camera prim: {ARGS.camera_prim}")
else:
    print(f"[INFO] Creating Camera prim: {ARGS.camera_prim}")

camera = Camera(
    prim_path=ARGS.camera_prim,
    name="main_camera_calibrated",
    frequency=FPS,
    resolution=(WIDTH, HEIGHT),
)

# Create/wrap render product first, then explicitly apply the USD-axis pose.
camera.initialize()

camera.set_world_pose(
    position=POSITION,
    orientation=QUAT_WXYZ,
    camera_axes="usd",
)

camera.set_resolution((WIDTH, HEIGHT))
camera.set_clipping_range(float(ARGS.near_clip), float(ARGS.far_clip))

# Use the exact calibrated pixel K. For inverse Brown, the simulation preview is
# intentionally undistorted; do not feed inverse coefficients as forward Brown.
camera.set_opencv_pinhole_properties(
    cx=CX,
    cy=CY,
    fx=FX,
    fy=FY,
    pinhole=PINHOLE_COEFFS.tolist(),
)

# Render several frames so USD/renderer state settles.
for _ in range(20):
    simulation_app.update()

actual_pos, actual_quat = camera.get_world_pose(camera_axes="usd")
actual_pos = np.asarray(actual_pos, dtype=np.float64).reshape(3)
actual_quat = np.asarray(actual_quat, dtype=np.float64).reshape(4)

# q and -q are the same orientation; compare with sign alignment.
quat_cmp = actual_quat.copy()
if float(np.dot(quat_cmp, QUAT_WXYZ)) < 0.0:
    quat_cmp *= -1.0

position_error = float(np.linalg.norm(actual_pos - POSITION))
quaternion_l2_error = float(np.linalg.norm(quat_cmp - QUAT_WXYZ))

print("\n[Pose verification]")
print(
    "Requested xyz [m] : "
    f"[{POSITION[0]:.6f}, {POSITION[1]:.6f}, {POSITION[2]:.6f}]"
)
print(
    "Isaac readback [m]: "
    f"[{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]"
)
print(f"Position error     : {position_error:.3e} m")
print(
    "Requested q wxyz  : "
    f"[{QUAT_WXYZ[0]:.7f}, {QUAT_WXYZ[1]:.7f}, "
    f"{QUAT_WXYZ[2]:.7f}, {QUAT_WXYZ[3]:.7f}]"
)
print(
    "Isaac readback q  : "
    f"[{actual_quat[0]:.7f}, {actual_quat[1]:.7f}, "
    f"{actual_quat[2]:.7f}, {actual_quat[3]:.7f}]"
)
print(f"Quaternion L2 err  : {quaternion_l2_error:.3e}")

try:
    cx_read, cy_read, fx_read, fy_read, coeffs_read = (
        camera.get_opencv_pinhole_properties()
    )
    print("\n[Intrinsic verification]")
    print(
        f"Isaac pinhole K    : fx={fx_read:.4f}, fy={fy_read:.4f}, "
        f"cx={cx_read:.4f}, cy={cy_read:.4f}"
    )
    print(
        "K abs error        : "
        f"fx={abs(float(fx_read)-FX):.3e}, "
        f"fy={abs(float(fy_read)-FY):.3e}, "
        f"cx={abs(float(cx_read)-CX):.3e}, "
        f"cy={abs(float(cy_read)-CY):.3e}"
    )
except Exception as exc:
    print(f"[WARN] Could not read back OpenCV pinhole properties: {exc}")

switch_active_viewport_to_camera(
    ARGS.camera_prim,
    (WIDTH, HEIGHT),
)

for _ in range(10):
    simulation_app.update()

print("\n" + "=" * 72)
print("Main camera is now placed from main_camera.yaml.")
print("The active viewport should show the scene through the calibrated camera.")
print("")
print("What to check visually:")
print("  1. Table/Piper framing matches the real D435i RGB view.")
print("  2. Left/right orientation is not mirrored.")
print("  3. Camera is above the table and looking down toward the workspace.")
print("  4. Board/workspace center appears in the expected image region.")
print("")
print(
    "This script changes only the in-memory stage. "
    "It does NOT overwrite the source testbed USD."
)
print("Close the Isaac Sim window or press Ctrl+C in the terminal to exit.")
print("=" * 72 + "\n")


try:
    while simulation_app.is_running():
        simulation_app.update()
except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C received.")
finally:
    simulation_app.close()
