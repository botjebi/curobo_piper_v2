#!/usr/bin/env python3
"""
RealSense RGB + ChArUco extrinsic calibration for the Piper/Isaac Sim workspace.

Purpose
-------
Estimate the 6D pose of the RealSense COLOR optical camera in the already-defined
Piper/Isaac Sim World frame, then save a canonical camera config that can later be
loaded by the cuRobo pick-and-place script.

This script intentionally does NOT create a new workspace frame from ChArUco and
it does NOT use depth or point clouds. The ChArUco board is a known world-frame
fiducial.

Fixed default board/world definition for this project
-----------------------------------------------------
ChArUco board:
    11 x 8 squares
    square length = 15 mm
    marker length = 12 mm
    dictionary = DICT_5X5_1000
    legacy pattern = True

Geometric board-center pose in Piper World:
    center = [0.345, 0.000, 0.000] m
    board +X -> world -X
    board +Y -> world -Y
    board +Z -> world +Z

Transform convention
--------------------
For T_A_B:
    p_A = T_A_B @ p_B

Frames:
    B : OpenCV CharucoBoard native planar frame used by solvePnP
    M : Physical ChArUco geometric-center frame used by Piper World
        +X follows the user's board +X direction
        +Y follows the user's board +Y direction
        +Z is the outward tabletop normal (+World Z)

    IMPORTANT:
    OpenCV's planar board convention places a front-view camera on the -Z side
    of the native board plane.  Therefore B and M are NOT axis-aligned here.
    The fixed B -> M orientation is Rx(pi):
        B +X -> M +X
        B +Y -> M -Y
        B +Z -> M -Z
    C : RealSense COLOR optical frame / OpenCV camera frame
        +X right, +Y down, +Z forward
    U : USD/Isaac Camera local frame
        +X right, +Y up, -Z forward
    W : Piper / Isaac Sim World frame

PnP returns T_C_B. The native board frame B is first converted to the
physical centered frame M with Rx(pi), then M is placed in Piper World.
Therefore:
    T_M_B = [Rx(pi), recenter translation]
    T_W_B = T_W_M @ T_M_B
    T_W_C = T_W_B @ inv(T_C_B)

The fixed camera-axis conversion is:
    T_C_U = diag(1, -1, -1, 1)
    T_W_U = T_W_C @ T_C_U

T_W_U is the transform intended for a USD camera prim.

UI
--
Capture + Calibrate button (or C key):
    1. Capture one RealSense RGB frame.
    2. Detect and visualize ArUco markers + ChArUco corners.
    3. Estimate board pose with solvePnPRansac + solvePnPRefineLM.
    4. Validate reprojection RMSE and pose sanity.
    5. Save configs/camera/main_camera.yaml on ACCEPTED results only.
    6. Save timestamped YAML + raw/annotated PNGs under logs/camera_calibration/.

Reconnect button:
    Restarts the RealSense pipeline and reloads intrinsics from the active COLOR
    stream profile.

Run
---
Recommended repository location:
    <PROJECT_ROOT>/src/calibration/realsense_charuco_camera_calibration.py

Standalone Isaac Sim Python:
    omni_python src/calibration/realsense_charuco_camera_calibration.py

Example profile override:
    omni_python src/calibration/realsense_charuco_camera_calibration.py \
        --color-width 1280 --color-height 720 --color-fps 30

Dependencies in the Isaac Sim Python environment:
    pyrealsense2
    PyYAML
    OpenCV with cv2.aruco / ChArUco support
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:  # handled before controller construction too
    rs = None  # type: ignore[assignment]
    _REALSENSE_IMPORT_ERROR = exc
else:
    _REALSENSE_IMPORT_ERROR = None

try:
    import yaml
except Exception as exc:
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Project defaults
# -----------------------------------------------------------------------------
DEFAULT_COLOR_WIDTH = 1280
DEFAULT_COLOR_HEIGHT = 720
DEFAULT_COLOR_FPS = 15
DEFAULT_CAPTURE_TIMEOUT_MS = 5000
DEFAULT_WARMUP_FRAMES = 15

DEFAULT_SQUARES_X = 11
DEFAULT_SQUARES_Y = 8
DEFAULT_SQUARE_LENGTH_M = 0.015
DEFAULT_MARKER_LENGTH_M = 0.012
DEFAULT_ARUCO_DICTIONARY = "DICT_5X5_1000"
DEFAULT_LEGACY_PATTERN = True

DEFAULT_BOARD_CENTER_WORLD_M = [0.345, 0.0, 0.0]
DEFAULT_BOARD_X_AXIS_WORLD = [1.0, 0.0, 0.0]
DEFAULT_BOARD_Y_AXIS_WORLD = [0.0, 1.0, 0.0]

DEFAULT_MIN_CHARUCO_CORNERS = 20
DEFAULT_MIN_PNP_INLIERS = 12
DEFAULT_MAX_REPROJECTION_RMSE_PX = 2.5
DEFAULT_PNP_RANSAC_REPROJECTION_PX = 2.0

DEFAULT_PREVIEW_MAX_WIDTH = 1280
DEFAULT_PREVIEW_MAX_HEIGHT = 720

CANONICAL_CONFIG_REL = Path("configs") / "camera" / "main_camera.yaml"
LOG_DIR_REL = Path("logs") / "camera_calibration"


# -----------------------------------------------------------------------------
# Project-root resolution
# -----------------------------------------------------------------------------
def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []

    if "__file__" in globals():
        try:
            file_parent = Path(__file__).resolve().parent
            candidates.extend([file_parent, *file_parent.parents])
        except Exception:
            pass

    try:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_project_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    for candidate in _candidate_roots():
        # curobo_piper_v2 has assets/ + configs/. Keep the test generic enough for
        # repository clones whose directory name may differ.
        if (candidate / "configs").is_dir() and (candidate / "assets").is_dir():
            return candidate

    # Fallback that matches the recommended <root>/src/calibration/<script>.py.
    if "__file__" in globals():
        p = Path(__file__).resolve().parent
        if len(p.parents) >= 2:
            return p.parents[1]

    return Path.cwd().resolve()


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a RealSense COLOR camera into the existing Piper/Isaac World "
            "frame using a known-pose ChArUco board."
        )
    )

    parser.add_argument("--project-root", default=None)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--color-width", type=int, default=DEFAULT_COLOR_WIDTH)
    parser.add_argument("--color-height", type=int, default=DEFAULT_COLOR_HEIGHT)
    parser.add_argument("--color-fps", type=int, default=DEFAULT_COLOR_FPS)
    parser.add_argument("--capture-timeout-ms", type=int, default=DEFAULT_CAPTURE_TIMEOUT_MS)
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)

    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    parser.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_M)
    parser.add_argument("--aruco-dictionary", default=DEFAULT_ARUCO_DICTIONARY)
    parser.add_argument(
        "--no-legacy-pattern",
        dest="legacy_pattern",
        action="store_false",
        help="Disable legacy ChArUco layout. Keep legacy enabled for the project board.",
    )
    parser.set_defaults(legacy_pattern=DEFAULT_LEGACY_PATTERN)

    parser.add_argument(
        "--board-center-world",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_BOARD_CENTER_WORLD_M,
        help="Geometric ChArUco center in Piper World, meters.",
    )
    parser.add_argument(
        "--board-x-axis-world",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_BOARD_X_AXIS_WORLD,
        help="Direction of centered-board +X expressed in Piper World.",
    )
    parser.add_argument(
        "--board-y-axis-world",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_BOARD_Y_AXIS_WORLD,
        help="Direction of centered-board +Y expressed in Piper World.",
    )

    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    parser.add_argument("--min-pnp-inliers", type=int, default=DEFAULT_MIN_PNP_INLIERS)
    parser.add_argument(
        "--max-reprojection-rmse",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_RMSE_PX,
    )
    parser.add_argument(
        "--pnp-ransac-reprojection",
        type=float,
        default=DEFAULT_PNP_RANSAC_REPROJECTION_PX,
    )

    parser.add_argument("--preview-max-width", type=int, default=DEFAULT_PREVIEW_MAX_WIDTH)
    parser.add_argument("--preview-max-height", type=int, default=DEFAULT_PREVIEW_MAX_HEIGHT)
    parser.add_argument(
        "--output-config",
        default=None,
        help="Canonical YAML path. Default: <project>/configs/camera/main_camera.yaml",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Calibration log directory. Default: <project>/logs/camera_calibration",
    )

    # Isaac/Kit appends its own CLI options.
    args, _unknown = parser.parse_known_args()

    if args.color_width <= 0 or args.color_height <= 0 or args.color_fps <= 0:
        parser.error("COLOR width/height/fps must be positive.")
    if args.capture_timeout_ms <= 0:
        parser.error("--capture-timeout-ms must be positive.")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be >= 0.")
    if args.squares_x < 2 or args.squares_y < 2:
        parser.error("ChArUco board must have at least 2 squares per axis.")
    if args.square_length <= 0 or args.marker_length <= 0:
        parser.error("Board lengths must be positive.")
    if args.marker_length >= args.square_length:
        parser.error("--marker-length must be smaller than --square-length.")
    if args.min_charuco_corners < 4:
        parser.error("--min-charuco-corners must be >= 4.")
    if args.min_pnp_inliers < 4:
        parser.error("--min-pnp-inliers must be >= 4.")
    if args.max_reprojection_rmse <= 0 or args.pnp_ransac_reprojection <= 0:
        parser.error("PnP/RMSE thresholds must be positive.")

    return args


ARGS = parse_args()
PROJECT_ROOT = find_project_root(ARGS.project_root)
CANONICAL_CONFIG_PATH = (
    Path(ARGS.output_config).expanduser().resolve()
    if ARGS.output_config
    else (PROJECT_ROOT / CANONICAL_CONFIG_REL).resolve()
)
LOG_DIR = (
    Path(ARGS.log_dir).expanduser().resolve()
    if ARGS.log_dir
    else (PROJECT_ROOT / LOG_DIR_REL).resolve()
)


# -----------------------------------------------------------------------------
# Start Isaac Sim only when not already running inside Kit.
# -----------------------------------------------------------------------------
SIMULATION_APP = None
RUNNING_INSIDE_KIT = False

try:
    import omni.kit.app  # type: ignore

    RUNNING_INSIDE_KIT = omni.kit.app.get_app() is not None
except Exception:
    RUNNING_INSIDE_KIT = False

if not RUNNING_INSIDE_KIT:
    try:
        from isaacsim import SimulationApp  # type: ignore
    except ImportError:
        try:
            from isaacsim.simulation_app import SimulationApp  # type: ignore
        except ImportError:
            from omni.isaac.kit import SimulationApp  # type: ignore

    SIMULATION_APP = SimulationApp({"headless": False, "width": 1600, "height": 900})


# Imports requiring a running Kit app.
import carb.input  # type: ignore  # noqa: E402
import omni.appwindow  # type: ignore  # noqa: E402
import omni.kit.app  # type: ignore  # noqa: E402
import omni.ui as ui  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fps: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs_rs: np.ndarray
    dist_coeffs_cv: np.ndarray
    pnp_distortion_handling: str
    rs_intrinsics: Any

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class PoseDetectionResult:
    success: bool
    message: str
    annotated_bgr: np.ndarray
    charuco_corner_count: int
    marker_count: int
    pnp_inlier_count: int = 0
    reprojection_rmse_px: Optional[float] = None
    rvec_camera_board: Optional[np.ndarray] = None
    tvec_camera_board_m: Optional[np.ndarray] = None
    T_camera_board_m: Optional[np.ndarray] = None


@dataclass
class CalibrationResult:
    timestamp_utc: str
    device_name: str
    serial_number: str
    firmware_version: str
    intrinsics: CameraIntrinsics
    center_board_native_m: np.ndarray
    T_world_board_center_m: np.ndarray
    T_center_board_native_m: np.ndarray
    T_world_board_native_m: np.ndarray
    T_camera_board_native_m: np.ndarray
    T_world_color_optical_m: np.ndarray
    T_world_isaac_camera_m: np.ndarray
    isaac_quaternion_wxyz: np.ndarray
    optical_quaternion_wxyz: np.ndarray
    charuco_corner_count: int
    marker_count: int
    pnp_inlier_count: int
    reprojection_rmse_px: float
    camera_height_world_m: float
    optical_forward_dot_to_board: float


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Cannot normalize {name}: {v}")
    return v / norm


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    T = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_is_valid(rotation: np.ndarray, atol: float = 1e-5) -> bool:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return bool(
        np.all(np.isfinite(R))
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 column-vector rotation matrix to [w, x, y, z]."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    # Project tiny numerical drift back onto SO(3).
    u, _s, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vh

    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
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
    # q and -q are equivalent. Keep a deterministic scalar-positive representation.
    if q[0] < 0.0:
        q = -q
    return q


def build_world_board_center_transform(args: argparse.Namespace) -> np.ndarray:
    x_world = normalize_vector(np.asarray(args.board_x_axis_world), "board +X in world")
    y_requested = normalize_vector(np.asarray(args.board_y_axis_world), "board +Y in world")

    # Orthogonalize Y against X and derive right-handed Z.
    y_world = y_requested - x_world * float(np.dot(x_world, y_requested))
    y_world = normalize_vector(y_world, "orthogonalized board +Y in world")
    z_world = normalize_vector(np.cross(x_world, y_world), "board +Z in world")
    y_world = normalize_vector(np.cross(z_world, x_world), "recomputed board +Y in world")

    R_world_center = np.column_stack((x_world, y_world, z_world))
    if not rotation_is_valid(R_world_center):
        raise RuntimeError(f"Invalid board/world rotation:\n{R_world_center}")

    return make_transform(R_world_center, np.asarray(args.board_center_world, dtype=np.float64))


def resize_for_preview(image_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return np.ascontiguousarray(image_bgr)
    return cv2.resize(
        image_bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


# -----------------------------------------------------------------------------
# RealSense helpers
# -----------------------------------------------------------------------------
def _rs_info(device: Any, info_enum: Any, fallback: str = "unknown") -> str:
    try:
        if device.supports(info_enum):
            return str(device.get_info(info_enum))
    except Exception:
        pass
    return fallback


def normalize_rs_distortion_name(model: Any) -> str:
    text = str(model).strip().lower()
    if "." in text:
        text = text.split(".")[-1]
    return text


def realsense_coeffs_to_opencv(
    intr: Any,
) -> tuple[str, np.ndarray, np.ndarray, str]:
    """
    Return (model_name, raw_rs_coeffs, cv_dist_coeffs, pnp_distortion_handling).

    OpenCV solvePnP accepts the forward Brown-Conrady coefficient convention.
    RealSense ``inverse_brown_conrady`` instead provides the closed-form mapping
    from a distorted pixel to an undistorted camera ray. For that model we do
    NOT pass the inverse coefficients to OpenCV. ChArUco image points are first
    converted to ideal pinhole pixels with
    ``rs.rs2_deproject_pixel_to_point(..., depth=1)`` and PnP then runs with the
    physical K matrix and zero OpenCV distortion.
    """
    model_name = normalize_rs_distortion_name(intr.model)
    raw = np.asarray(list(intr.coeffs), dtype=np.float64).reshape(-1)
    if raw.size < 5:
        raw = np.pad(raw, (0, 5 - raw.size))
    raw5 = raw[:5].copy()

    if model_name == "none":
        return (
            model_name,
            raw5,
            np.zeros(5, dtype=np.float64),
            "opencv_pinhole_no_distortion",
        )

    if model_name in {"brown_conrady", "modified_brown_conrady"}:
        return (
            model_name,
            raw5,
            raw5.copy(),
            "opencv_forward_brown_conrady",
        )

    if model_name == "inverse_brown_conrady":
        return (
            model_name,
            raw5,
            np.zeros(5, dtype=np.float64),
            "realsense_deproject_to_pinhole",
        )

    raise RuntimeError(
        "Unsupported RealSense COLOR distortion model "
        f"'{model_name}'. Supported models: none, brown_conrady, "
        "modified_brown_conrady, inverse_brown_conrady."
    )


# -----------------------------------------------------------------------------
# ChArUco pose estimator
# -----------------------------------------------------------------------------
class CharucoPoseEstimator:
    def __init__(self, args: argparse.Namespace, intrinsics: CameraIntrinsics) -> None:
        self.args = args
        self.intrinsics = intrinsics

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without cv2.aruco support.")
        if not hasattr(cv2.aruco, args.aruco_dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {args.aruco_dictionary}")
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise RuntimeError(
                "This utility expects the modern OpenCV ChArUco API (cv2.aruco.CharucoDetector)."
            )

        dictionary_id = getattr(cv2.aruco, args.aruco_dictionary)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            float(args.square_length),
            float(args.marker_length),
            self.dictionary,
        )
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(bool(args.legacy_pattern))

        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_parameters.cameraMatrix = self.intrinsics.K
        charuco_parameters.distCoeffs = self.intrinsics.dist_coeffs_cv
        charuco_parameters.tryRefineMarkers = True

        refine_parameters = cv2.aruco.RefineParameters()
        self.detector = cv2.aruco.CharucoDetector(
            self.board,
            charuco_parameters,
            detector_parameters,
            refine_parameters,
        )

        chessboard_corners = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3)
        # Compute totals from the actual OpenCV board object rather than
        # hard-coding 44 markers / 70 ChArUco corners. This keeps the UI correct
        # if the board definition is changed later.
        self.total_charuco_corner_count = int(len(chessboard_corners))

        board_ids = self.board.getIds()
        self.total_marker_count = 0 if board_ids is None else int(len(board_ids))

        # For this symmetric checkerboard, the mean of the internal-corner grid is
        # exactly the geometric center of the complete printed checker region.
        self.center_board_native_m = chessboard_corners.mean(axis=0)

    @staticmethod
    def _count_with_ratio(count: int, total: int) -> str:
        if total <= 0:
            return f"{count}/?"
        ratio = 100.0 * float(count) / float(total)
        return f"{count}/{total} ({ratio:.1f}%)"

    @staticmethod
    def _draw_status_text(image: np.ndarray, lines: list[str], accepted: bool) -> None:
        color = (0, 220, 0) if accepted else (0, 0, 255)
        line_height = 30
        box_h = 15 + line_height * len(lines)
        cv2.rectangle(image, (8, 8), (min(image.shape[1] - 8, 1200), box_h), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (16, 36 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color if i == 0 else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _prepare_pnp_image_points(
        self,
        image_points_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raw detected pixels to the image model expected by solvePnP.

        For inverse Brown-Conrady, RealSense supplies a direct distorted-pixel
        -> undistorted-ray mapping through rs2_deproject_pixel_to_point().
        """
        points = np.ascontiguousarray(
            image_points_raw, dtype=np.float64
        ).reshape(-1, 2)

        if self.intrinsics.pnp_distortion_handling != "realsense_deproject_to_pinhole":
            return points, self.intrinsics.dist_coeffs_cv

        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is unavailable while inverse Brown-Conrady "
                "deprojection is required."
            )

        ideal_pixels = np.empty_like(points, dtype=np.float64)
        for i, (u, v) in enumerate(points):
            ray = rs.rs2_deproject_pixel_to_point(
                self.intrinsics.rs_intrinsics,
                [float(u), float(v)],
                1.0,
            )
            X, Y, Z = [float(value) for value in ray]
            if not np.all(np.isfinite([X, Y, Z])) or abs(Z) < 1e-12:
                raise RuntimeError(
                    f"Invalid RealSense deprojection for pixel {(u, v)}: {ray}"
                )

            x = X / Z
            y = Y / Z
            ideal_pixels[i, 0] = self.intrinsics.fx * x + self.intrinsics.cx
            ideal_pixels[i, 1] = self.intrinsics.fy * y + self.intrinsics.cy

        return (
            np.ascontiguousarray(ideal_pixels, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
        )

    def detect_and_estimate(self, image_bgr: np.ndarray) -> PoseDetectionResult:
        image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        annotated = image_bgr.copy()
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        corner_count = 0 if charuco_ids is None else int(len(charuco_ids))

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated, charuco_corners, charuco_ids, cornerColor=(0, 255, 255)
            )

        if corner_count < self.args.min_charuco_corners:
            message = (
                f"REJECTED: {corner_count} ChArUco corners; "
                f"need >= {self.args.min_charuco_corners}."
            )
            self._draw_status_text(
                annotated,
                [
                    message,
                    (
                        "ArUco markers="
                        f"{self._count_with_ratio(marker_count, self.total_marker_count)}"
                    ),
                    (
                        "ChArUco corners="
                        f"{self._count_with_ratio(corner_count, self.total_charuco_corner_count)}"
                    ),
                ],
                accepted=False,
            )
            return PoseDetectionResult(
                success=False,
                message=message,
                annotated_bgr=annotated,
                charuco_corner_count=corner_count,
                marker_count=marker_count,
            )

        object_points, image_points_raw = self.board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        object_points = np.ascontiguousarray(
            object_points, dtype=np.float64
        ).reshape(-1, 3)
        image_points_raw = np.ascontiguousarray(
            image_points_raw, dtype=np.float64
        ).reshape(-1, 2)

        image_points_pnp, pnp_dist_coeffs = self._prepare_pnp_image_points(
            image_points_raw
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points_pnp,
            self.intrinsics.K,
            pnp_dist_coeffs,
            iterationsCount=300,
            reprojectionError=float(self.args.pnp_ransac_reprojection),
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        inlier_count = 0 if inliers is None else int(len(inliers))
        if not success or rvec is None or tvec is None:
            message = "REJECTED: solvePnPRansac failed."
            self._draw_status_text(
                annotated,
                [
                    message,
                    (
                        "ArUco markers="
                        f"{self._count_with_ratio(marker_count, self.total_marker_count)}"
                    ),
                    (
                        "ChArUco corners="
                        f"{self._count_with_ratio(corner_count, self.total_charuco_corner_count)}"
                    ),
                ],
                accepted=False,
            )
            return PoseDetectionResult(
                success=False,
                message=message,
                annotated_bgr=annotated,
                charuco_corner_count=corner_count,
                marker_count=marker_count,
                pnp_inlier_count=inlier_count,
            )

        if inlier_count < self.args.min_pnp_inliers:
            message = (
                f"REJECTED: {inlier_count} PnP inliers; "
                f"need >= {self.args.min_pnp_inliers}."
            )
            self._draw_status_text(
                annotated,
                [
                    message,
                    (
                        "ArUco markers="
                        f"{self._count_with_ratio(marker_count, self.total_marker_count)}"
                    ),
                    (
                        "ChArUco corners="
                        f"{self._count_with_ratio(corner_count, self.total_charuco_corner_count)}"
                    ),
                    (
                        f"PnP inliers={inlier_count}/{corner_count} "
                        f"({(100.0 * inlier_count / corner_count) if corner_count else 0.0:.1f}%)"
                    ),
                ],
                accepted=False,
            )
            return PoseDetectionResult(
                success=False,
                message=message,
                annotated_bgr=annotated,
                charuco_corner_count=corner_count,
                marker_count=marker_count,
                pnp_inlier_count=inlier_count,
            )

        # Refine using RANSAC inliers only.
        try:
            idx = inliers.reshape(-1)
            rvec, tvec = cv2.solvePnPRefineLM(
                np.ascontiguousarray(object_points[idx], dtype=np.float64),
                np.ascontiguousarray(image_points_pnp[idx], dtype=np.float64),
                self.intrinsics.K,
                pnp_dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.intrinsics.K,
            pnp_dist_coeffs,
        )
        residual = projected.reshape(-1, 2) - image_points_pnp
        reprojection_rmse_px = float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        )

        R_camera_board, _ = cv2.Rodrigues(rvec)
        T_camera_board = make_transform(
            R_camera_board, np.asarray(tvec).reshape(3)
        )

        # drawFrameAxes is exact on the raw image only when OpenCV owns the
        # forward distortion model. For inverse Brown-Conrady the PnP points are
        # converted to an ideal pinhole image first, so keep marker/corner
        # overlays and status text on the raw preview rather than draw a
        # potentially misleading axis overlay.
        if self.intrinsics.pnp_distortion_handling != "realsense_deproject_to_pinhole":
            center_in_camera = (
                R_camera_board @ self.center_board_native_m
                + np.asarray(tvec).reshape(3)
            )
            cv2.drawFrameAxes(
                annotated,
                self.intrinsics.K,
                pnp_dist_coeffs,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                center_in_camera.reshape(3, 1),
                float(self.args.square_length) * 4.0,
                3,
            )

        accepted = reprojection_rmse_px <= float(self.args.max_reprojection_rmse)
        if not accepted:
            message = (
                f"REJECTED: reprojection RMSE {reprojection_rmse_px:.3f}px > "
                f"{self.args.max_reprojection_rmse:.3f}px."
            )
        else:
            message = (
                "ACCEPTED: "
                f"markers={self._count_with_ratio(marker_count, self.total_marker_count)}, "
                f"corners={self._count_with_ratio(corner_count, self.total_charuco_corner_count)}, "
                f"inliers={inlier_count}/{corner_count}, "
                f"RMSE={reprojection_rmse_px:.3f}px"
            )

        self._draw_status_text(
            annotated,
            [
                message,
                (
                    "ArUco markers="
                    f"{self._count_with_ratio(marker_count, self.total_marker_count)}"
                ),
                (
                    "ChArUco corners="
                    f"{self._count_with_ratio(corner_count, self.total_charuco_corner_count)}"
                ),
                (
                    f"PnP inliers={inlier_count}/{corner_count} "
                    f"({(100.0 * inlier_count / corner_count) if corner_count else 0.0:.1f}%)"
                ),
                f"distortion handling={self.intrinsics.pnp_distortion_handling}",
            ],
            accepted=accepted,
        )

        return PoseDetectionResult(
            success=accepted,
            message=message,
            annotated_bgr=annotated,
            charuco_corner_count=corner_count,
            marker_count=marker_count,
            pnp_inlier_count=inlier_count,
            reprojection_rmse_px=reprojection_rmse_px,
            rvec_camera_board=np.asarray(rvec, dtype=np.float64).reshape(3),
            tvec_camera_board_m=np.asarray(tvec, dtype=np.float64).reshape(3),
            T_camera_board_m=T_camera_board,
        )


# -----------------------------------------------------------------------------
# Calibration construction / validation
# -----------------------------------------------------------------------------
def construct_calibration(
    args: argparse.Namespace,
    detection: PoseDetectionResult,
    estimator: CharucoPoseEstimator,
    intrinsics: CameraIntrinsics,
    device_name: str,
    serial_number: str,
    firmware_version: str,
) -> CalibrationResult:
    if not detection.success or detection.T_camera_board_m is None:
        raise ValueError("Cannot construct calibration from a rejected ChArUco detection.")
    if detection.reprojection_rmse_px is None:
        raise ValueError("Missing reprojection RMSE.")

    T_world_center = build_world_board_center_transform(args)

    center_B = np.asarray(
        estimator.center_board_native_m, dtype=np.float64
    ).reshape(3)

    # OpenCV's planar ChArUco/object-point convention and the physical board
    # frame used by this Piper workspace differ by 180 deg about board X.
    #
    # Physical centered frame M:
    #   +X = user's board +X
    #   +Y = user's board +Y
    #   +Z = outward tabletop normal (+World Z)
    #
    # OpenCV native planar frame B used by PnP:
    #   B +X -> M +X
    #   B +Y -> M -Y
    #   B +Z -> M -Z
    #
    # Therefore:
    #   p_M = R_M_B @ (p_B - center_B)
    R_center_board_native = np.diag([1.0, -1.0, -1.0])
    t_center_board_native = -R_center_board_native @ center_B
    T_center_board_native = make_transform(
        R_center_board_native,
        t_center_board_native,
    )
    T_world_board_native = T_world_center @ T_center_board_native

    T_camera_board = np.asarray(detection.T_camera_board_m, dtype=np.float64)
    T_world_color_optical = T_world_board_native @ invert_rigid_transform(T_camera_board)

    # OpenCV optical C: +X right, +Y down, +Z forward
    # USD camera U:      +X right, +Y up,   -Z forward
    T_color_optical_isaac_camera = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    T_world_isaac_camera = T_world_color_optical @ T_color_optical_isaac_camera

    if not rotation_is_valid(T_world_color_optical[:3, :3]):
        raise RuntimeError("T_world_color_optical has an invalid rotation.")
    if not rotation_is_valid(T_world_isaac_camera[:3, :3]):
        raise RuntimeError("T_world_isaac_camera has an invalid rotation.")

    camera_pos_world = T_world_color_optical[:3, 3]
    board_center_world = T_world_center[:3, 3]
    camera_height_world_m = float(camera_pos_world[2])

    to_board = board_center_world - camera_pos_world
    to_board_norm = float(np.linalg.norm(to_board))
    if to_board_norm < 1e-9:
        raise RuntimeError("Camera position is numerically at the board center.")
    to_board_unit = to_board / to_board_norm
    optical_forward_world = T_world_color_optical[:3, 2]
    optical_forward_dot_to_board = float(np.dot(optical_forward_world, to_board_unit))

    # Project-specific sanity checks. The board is on the tabletop and the main
    # camera should be above it, looking toward it. These checks also reject the
    # common planar-PnP flipped-pose failure mode.
    if camera_height_world_m <= float(args.board_center_world[2]):
        raise RuntimeError(
            "Pose sanity check failed: estimated camera is not above the board/table. "
            f"camera_z={camera_height_world_m:.4f} m"
        )
    if optical_forward_dot_to_board <= 0.0:
        raise RuntimeError(
            "Pose sanity check failed: RealSense optical +Z does not point toward the board. "
            f"dot={optical_forward_dot_to_board:.4f}"
        )

    return CalibrationResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        device_name=device_name,
        serial_number=serial_number,
        firmware_version=firmware_version,
        intrinsics=intrinsics,
        center_board_native_m=center_B,
        T_world_board_center_m=T_world_center,
        T_center_board_native_m=T_center_board_native,
        T_world_board_native_m=T_world_board_native,
        T_camera_board_native_m=T_camera_board,
        T_world_color_optical_m=T_world_color_optical,
        T_world_isaac_camera_m=T_world_isaac_camera,
        isaac_quaternion_wxyz=rotation_matrix_to_quaternion_wxyz(T_world_isaac_camera[:3, :3]),
        optical_quaternion_wxyz=rotation_matrix_to_quaternion_wxyz(T_world_color_optical[:3, :3]),
        charuco_corner_count=detection.charuco_corner_count,
        marker_count=detection.marker_count,
        pnp_inlier_count=detection.pnp_inlier_count,
        reprojection_rmse_px=float(detection.reprojection_rmse_px),
        camera_height_world_m=camera_height_world_m,
        optical_forward_dot_to_board=optical_forward_dot_to_board,
    )


def annotate_world_pose(
    image_bgr: np.ndarray,
    calibration: CalibrationResult,
) -> np.ndarray:
    out = image_bgr.copy()
    p = calibration.T_world_isaac_camera_m[:3, 3]
    q = calibration.isaac_quaternion_wxyz
    lines = [
        "Piper World camera pose saved",
        f"USD xyz [m] = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})",
        f"USD quat wxyz = ({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})",
        f"optical-forward dot(to board) = {calibration.optical_forward_dot_to_board:.4f}",
    ]
    start_y = max(90, out.shape[0] - (len(lines) * 30 + 18))
    cv2.rectangle(
        out,
        (8, start_y - 30),
        (min(out.shape[1] - 8, 1250), out.shape[0] - 8),
        (0, 0, 0),
        -1,
    )
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (16, start_y + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255) if i else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def calibration_to_yaml_dict(
    calibration: CalibrationResult,
    args: argparse.Namespace,
    canonical_path: Path,
    log_yaml_path: Path,
    raw_path: Path,
    preview_path: Path,
) -> dict[str, Any]:
    intr = calibration.intrinsics
    T_w_c = calibration.T_world_color_optical_m
    T_w_u = calibration.T_world_isaac_camera_m

    fov_x_deg = math.degrees(2.0 * math.atan2(intr.width, 2.0 * intr.fx))
    fov_y_deg = math.degrees(2.0 * math.atan2(intr.height, 2.0 * intr.fy))

    return {
        "schema": "piper_realsense_charuco_camera_calibration",
        "schema_version": 1,
        "timestamp_utc": calibration.timestamp_utc,
        "transform_convention": {
            "rule": "p_A = T_A_B @ p_B",
            "vectors": "column",
            "translation_unit": "meter",
            "quaternion_order": "wxyz",
        },
        "camera": {
            "device_name": calibration.device_name,
            "serial_number": calibration.serial_number,
            "firmware_version": calibration.firmware_version,
            "stream": "color",
            "format": "bgr8",
            "width": intr.width,
            "height": intr.height,
            "fps": intr.fps,
        },
        "intrinsics": {
            "K": intr.K.tolist(),
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "distortion_model_realsense": intr.model,
            "distortion_coefficients_realsense": intr.coeffs_rs.tolist(),
            "distortion_coefficients_opencv_k1_k2_p1_p2_k3": intr.dist_coeffs_cv.tolist(),
            "pnp_distortion_handling": intr.pnp_distortion_handling,
            "fov_x_deg_from_pinhole_intrinsics": fov_x_deg,
            "fov_y_deg_from_pinhole_intrinsics": fov_y_deg,
        },
        "charuco": {
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_length_m": float(args.square_length),
            "marker_length_m": float(args.marker_length),
            "aruco_dictionary": str(args.aruco_dictionary),
            "legacy_pattern": bool(args.legacy_pattern),
            "geometric_center_in_native_board_m": calibration.center_board_native_m.tolist(),
            "native_to_physical_center_rotation": "Rx(pi) = diag(1,-1,-1)",
            "native_axis_mapping": {
                "B_plus_X": "M_plus_X",
                "B_plus_Y": "M_minus_Y",
                "B_plus_Z": "M_minus_Z",
            },
        },
        "board_pose_world": {
            "geometric_center_xyz_m": np.asarray(args.board_center_world, dtype=float).tolist(),
            "x_axis_world": calibration.T_world_board_center_m[:3, 0].tolist(),
            "y_axis_world": calibration.T_world_board_center_m[:3, 1].tolist(),
            "z_axis_world": calibration.T_world_board_center_m[:3, 2].tolist(),
            "T_world_board_center": calibration.T_world_board_center_m.tolist(),
            "T_center_board_native": calibration.T_center_board_native_m.tolist(),
            "T_world_board_native": calibration.T_world_board_native_m.tolist(),
        },
        "calibration": {
            "T_camera_optical_board_native": calibration.T_camera_board_native_m.tolist(),
            "T_world_color_optical_cv": T_w_c.tolist(),
            "T_world_isaac_camera_usd": T_w_u.tolist(),
            "color_optical_position_world_m": T_w_c[:3, 3].tolist(),
            "color_optical_quaternion_world_wxyz": calibration.optical_quaternion_wxyz.tolist(),
            "isaac_camera_position_world_m": T_w_u[:3, 3].tolist(),
            "isaac_camera_quaternion_world_wxyz": calibration.isaac_quaternion_wxyz.tolist(),
        },
        "quality": {
            "charuco_corner_count": calibration.charuco_corner_count,
            "charuco_corner_total": int(
                (int(args.squares_x) - 1) * (int(args.squares_y) - 1)
            ),
            "charuco_corner_detection_ratio": float(
                calibration.charuco_corner_count
                / max(1, (int(args.squares_x) - 1) * (int(args.squares_y) - 1))
            ),
            "aruco_marker_count": calibration.marker_count,
            "aruco_marker_total": int(
                ((int(args.squares_x) * int(args.squares_y)) + 1) // 2
                if bool(args.legacy_pattern)
                else (int(args.squares_x) * int(args.squares_y)) // 2
            ),
            "pnp_inlier_count": calibration.pnp_inlier_count,
            "pnp_inlier_ratio_of_detected_corners": float(
                calibration.pnp_inlier_count / max(1, calibration.charuco_corner_count)
            ),
            "reprojection_rmse_px": calibration.reprojection_rmse_px,
            "max_reprojection_rmse_px": float(args.max_reprojection_rmse),
            "camera_height_world_m": calibration.camera_height_world_m,
            "optical_forward_dot_to_board": calibration.optical_forward_dot_to_board,
            "accepted": True,
        },
        "isaac_usage": {
            "pose_transform": "calibration.T_world_isaac_camera_usd",
            "position": "calibration.isaac_camera_position_world_m",
            "quaternion_wxyz": "calibration.isaac_camera_quaternion_world_wxyz",
            "note": (
                "These fields represent the USD camera local-axis convention: "
                "+X right, +Y up, -Z forward."
            ),
        },
        "files": {
            "canonical_config": str(canonical_path),
            "timestamped_log_yaml": str(log_yaml_path),
            "raw_rgb": str(raw_path),
            "annotated_preview": str(preview_path),
        },
    }


def atomic_yaml_dump(payload: dict[str, Any], path: Path) -> None:
    if yaml is None:
        raise RuntimeError(f"PyYAML import failed: {_YAML_IMPORT_ERROR}")
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def save_calibration_files(
    calibration: CalibrationResult,
    raw_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    CANONICAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"main_camera_charuco_{stamp}"
    log_yaml = LOG_DIR / f"{stem}.yaml"
    raw_path = LOG_DIR / f"{stem}_raw.png"
    preview_path = LOG_DIR / f"{stem}_preview.png"

    payload = calibration_to_yaml_dict(
        calibration,
        args,
        CANONICAL_CONFIG_PATH,
        log_yaml,
        raw_path,
        preview_path,
    )

    if not cv2.imwrite(str(raw_path), raw_bgr):
        raise IOError(f"Failed to save raw RGB image: {raw_path}")
    if not cv2.imwrite(str(preview_path), annotated_bgr):
        raise IOError(f"Failed to save annotated preview: {preview_path}")

    atomic_yaml_dump(payload, log_yaml)
    atomic_yaml_dump(payload, CANONICAL_CONFIG_PATH)

    # Convenience latest copies in logs while keeping canonical config in configs/.
    latest_log = LOG_DIR / "main_camera_charuco_latest.yaml"
    latest_raw = LOG_DIR / "main_camera_charuco_latest_raw.png"
    latest_preview = LOG_DIR / "main_camera_charuco_latest_preview.png"
    shutil.copy2(log_yaml, latest_log)
    shutil.copy2(raw_path, latest_raw)
    shutil.copy2(preview_path, latest_preview)

    return CANONICAL_CONFIG_PATH, log_yaml, raw_path, preview_path


# -----------------------------------------------------------------------------
# Main Isaac UI controller
# -----------------------------------------------------------------------------
class RealSenseCharucoCalibrationController:
    def __init__(self, args: argparse.Namespace) -> None:
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not importable in the Isaac Sim Python environment. "
                f"Original error: {_REALSENSE_IMPORT_ERROR}"
            )
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not importable in the Isaac Sim Python environment. "
                f"Original error: {_YAML_IMPORT_ERROR}"
            )

        self.args = args
        self.pipeline: Optional[Any] = None
        self.pipeline_profile: Optional[Any] = None
        self.connected = False
        self.busy = False

        self.device_name = "unknown"
        self.serial_number = "unknown"
        self.firmware_version = "unknown"
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.pose_estimator: Optional[CharucoPoseEstimator] = None
        self.last_calibration: Optional[CalibrationResult] = None
        self.last_saved_config: Optional[Path] = None

        self.window: Optional[Any] = None
        self.preview_window: Optional[Any] = None
        self.preview_provider: Optional[Any] = None
        self.preview_image: Optional[Any] = None
        self.preview_label: Optional[Any] = None
        self.camera_label: Optional[Any] = None
        self.detection_label: Optional[Any] = None
        self.pose_label: Optional[Any] = None
        self.file_label: Optional[Any] = None
        self.status_label: Optional[Any] = None

        self._preview_upload_task: Optional[Any] = None
        self._preview_generation = 0
        self._preview_texture_name = f"piper_realsense_charuco_preview_{os.getpid()}"
        self._preview_rgba_format: Optional[Any] = None
        self._opencv_preview_fallback = False

        self._build_ui()
        self._setup_keyboard()

        # Connect immediately so profile/intrinsics are visible before capture.
        self.connect_camera()
        if self.connected:
            self.set_status("Ready. Press Capture + Calibrate (or C).")
        else:
            self.set_status("Camera not connected. Use Reconnect, then capture.")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.window = ui.Window("Piper RealSense ChArUco Calibration", width=760, height=390)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Known board center W = (0.345, 0.000, 0.000) m")
                ui.Label("board +X -> World -X | board +Y -> World -Y | board +Z -> World +Z")
                ui.Separator()
                self.camera_label = ui.Label("Camera: initializing", word_wrap=True)
                self.detection_label = ui.Label("ChArUco detection: none", word_wrap=True)
                self.pose_label = ui.Label("World camera pose: none", word_wrap=True)
                self.file_label = ui.Label(f"Canonical config: {CANONICAL_CONFIG_PATH}", word_wrap=True)
                self.status_label = ui.Label("Status: starting", word_wrap=True)
                with ui.HStack(spacing=8, height=36):
                    ui.Button("Capture + Calibrate (C)", clicked_fn=self.capture_and_calibrate)
                    ui.Button("Reconnect Camera", clicked_fn=self.reconnect_camera)
                    ui.Button("Clear Preview", clicked_fn=self.clear_preview)

        self.preview_window = ui.Window(
            "RealSense RGB / ChArUco Detection",
            width=1120,
            height=820,
        )
        with self.preview_window.frame:
            with ui.VStack(spacing=6):
                with ui.HStack(spacing=8, height=34):
                    ui.Button("Capture + Calibrate (C)", clicked_fn=self.capture_and_calibrate)
                    ui.Button("Reconnect Camera", clicked_fn=self.reconnect_camera)
                self.preview_label = ui.Label("No RGB image captured yet.", height=24, word_wrap=True)
                try:
                    provider_cls = getattr(ui, "DynamicTextureProvider", None)
                    if provider_cls is not None:
                        self.preview_provider = provider_cls(self._preview_texture_name)
                    else:
                        self.preview_provider = ui.ByteImageProvider()

                    texture_format_cls = getattr(ui, "TextureFormat", None)
                    self._preview_rgba_format = (
                        getattr(texture_format_cls, "RGBA8_UNORM", None)
                        if texture_format_cls is not None
                        else None
                    )

                    self._upload_rgba_to_provider(
                        np.array(
                            [
                                [[32, 32, 32, 255], [64, 64, 64, 255]],
                                [[64, 64, 64, 255], [32, 32, 32, 255]],
                            ],
                            dtype=np.uint8,
                        )
                    )
                    self.preview_image = ui.ImageWithProvider(
                        self.preview_provider,
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER,
                    )
                except Exception as exc:
                    self.preview_provider = None
                    self.preview_image = None
                    self._opencv_preview_fallback = True
                    ui.Label(
                        "Omni UI dynamic image provider unavailable; OpenCV preview will be used. "
                        f"Details: {exc}",
                        word_wrap=True,
                    )

        self.preview_window.visible = False

    def _setup_keyboard(self) -> None:
        self.input_interface = carb.input.acquire_input_interface()
        self.app_window = omni.appwindow.get_default_app_window()
        if self.app_window is None:
            raise RuntimeError("Isaac Sim default app window is unavailable.")
        self.keyboard = self.app_window.get_keyboard()
        self.keyboard_sub_id = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event: Any) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.C:
            self.capture_and_calibrate()
            return True
        return True

    def _upload_rgba_to_provider(self, rgba: np.ndarray) -> None:
        if self.preview_provider is None:
            raise RuntimeError("Preview provider is not initialized.")
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Expected HxWx4 RGBA image, got {rgba.shape}")

        height, width = rgba.shape[:2]
        flat_array = np.ascontiguousarray(rgba.reshape(-1), dtype=np.uint8)
        flat_list = flat_array.tolist()
        stride = int(width * 4)

        errors: list[str] = []
        if self._preview_rgba_format is not None:
            for call in (
                lambda: self.preview_provider.set_bytes_data(
                    flat_list,
                    [int(width), int(height)],
                    self._preview_rgba_format,
                    stride,
                ),
                lambda: self.preview_provider.set_bytes_data(
                    flat_list,
                    [int(width), int(height)],
                    self._preview_rgba_format,
                ),
            ):
                try:
                    call()
                    return
                except Exception as exc:
                    errors.append(str(exc))

        for call in (
            lambda: self.preview_provider.set_data_array(flat_array, [int(width), int(height)]),
            lambda: self.preview_provider.set_bytes_data(flat_list, [int(width), int(height)]),
        ):
            try:
                call()
                return
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError("; ".join(errors[-4:]))

    async def _upload_preview_after_next_frame(self, rgba: np.ndarray, generation: int) -> None:
        try:
            await omni.kit.app.get_app().next_update_async()
            if generation != self._preview_generation:
                return
            self._upload_rgba_to_provider(rgba)
            await omni.kit.app.get_app().next_update_async()
        except Exception as exc:
            print(f"[WARN] Deferred Omni UI preview upload failed: {exc}")
            self._opencv_preview_fallback = True
            try:
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                cv2.imshow("RealSense RGB / ChArUco Detection", bgr)
                cv2.waitKey(1)
            except cv2.error as cv_exc:
                print(f"[WARN] OpenCV preview failed: {cv_exc}")

    def _update_preview(self, image_bgr: np.ndarray, label: str) -> None:
        image_bgr = resize_for_preview(
            image_bgr, self.args.preview_max_width, self.args.preview_max_height
        )
        if self.preview_label is not None:
            self.preview_label.text = label
        if self.preview_window is not None:
            self.preview_window.visible = True

        rgba = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA), dtype=np.uint8)

        if self.preview_provider is not None:
            self._preview_generation += 1
            generation = self._preview_generation
            try:
                loop = asyncio.get_event_loop()
                self._preview_upload_task = loop.create_task(
                    self._upload_preview_after_next_frame(rgba.copy(), generation)
                )
                return
            except Exception as exc:
                print(f"[WARN] Could not schedule preview upload: {exc}")
                try:
                    self._upload_rgba_to_provider(rgba)
                    return
                except Exception as direct_exc:
                    print(f"[WARN] Direct Omni UI preview failed: {direct_exc}")
                    self._opencv_preview_fallback = True

        if self._opencv_preview_fallback:
            try:
                cv2.imshow("RealSense RGB / ChArUco Detection", image_bgr)
                cv2.waitKey(1)
            except cv2.error as exc:
                print(f"[WARN] OpenCV preview failed: {exc}")

    def clear_preview(self) -> None:
        self._preview_generation += 1
        self.last_calibration = None
        if self.preview_label is not None:
            self.preview_label.text = "Preview cleared. Saved files were retained."
        if self.detection_label is not None:
            self.detection_label.text = "ChArUco detection: none"
        if self.pose_label is not None:
            self.pose_label.text = "World camera pose: none"
        if self.preview_provider is not None:
            try:
                self._upload_rgba_to_provider(
                    np.full((2, 2, 4), [48, 48, 48, 255], dtype=np.uint8)
                )
            except Exception:
                pass
        self.set_status("Preview cleared. Saved calibration remains unchanged.")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        print(f"[Piper Camera Calibration] {text}")
        if self.status_label is not None:
            self.status_label.text = f"Status: {text}"

    def update_camera_label(self) -> None:
        if self.camera_label is None:
            return
        if not self.connected or self.intrinsics is None:
            self.camera_label.text = "Camera: disconnected"
            return
        intr = self.intrinsics
        self.camera_label.text = (
            f"Camera: {self.device_name} | serial={self.serial_number} | "
            f"COLOR {intr.width}x{intr.height}@{intr.fps} | "
            f"fx={intr.fx:.2f}, fy={intr.fy:.2f} | distortion={intr.model}"
        )

    def update_detection_label(self, detection: Optional[PoseDetectionResult]) -> None:
        if self.detection_label is None:
            return
        if detection is None:
            self.detection_label.text = "ChArUco detection: none"
            return
        rmse = (
            f"{detection.reprojection_rmse_px:.3f}px"
            if detection.reprojection_rmse_px is not None
            else "n/a"
        )
        result = "ACCEPTED" if detection.success else "REJECTED"
        total_markers = (
            self.pose_estimator.total_marker_count
            if self.pose_estimator is not None
            else 0
        )
        total_corners = (
            self.pose_estimator.total_charuco_corner_count
            if self.pose_estimator is not None
            else 0
        )
        marker_text = CharucoPoseEstimator._count_with_ratio(
            detection.marker_count, total_markers
        )
        corner_text = CharucoPoseEstimator._count_with_ratio(
            detection.charuco_corner_count, total_corners
        )
        inlier_ratio = (
            100.0 * detection.pnp_inlier_count / detection.charuco_corner_count
            if detection.charuco_corner_count > 0
            else 0.0
        )
        self.detection_label.text = (
            f"Detection: ArUco markers={marker_text} | "
            f"ChArUco corners={corner_text} | "
            f"PnP inliers={detection.pnp_inlier_count}/"
            f"{detection.charuco_corner_count} ({inlier_ratio:.1f}%) | "
            f"RMSE={rmse} | {result}"
        )

    def update_pose_label(self, calibration: Optional[CalibrationResult]) -> None:
        if self.pose_label is None:
            return
        if calibration is None:
            self.pose_label.text = "World camera pose: none"
            return
        p = calibration.T_world_isaac_camera_m[:3, 3]
        q = calibration.isaac_quaternion_wxyz
        self.pose_label.text = (
            "World USD camera pose: "
            f"xyz=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m | "
            f"q_wxyz=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})"
        )

    # ------------------------------------------------------------------
    # RealSense connection
    # ------------------------------------------------------------------
    def connect_camera(self) -> bool:
        if self.connected:
            return True
        assert rs is not None

        self.set_status("Connecting to RealSense COLOR stream...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            if self.args.serial:
                config.enable_device(str(self.args.serial))
            config.enable_stream(
                rs.stream.color,
                int(self.args.color_width),
                int(self.args.color_height),
                rs.format.bgr8,
                int(self.args.color_fps),
            )

            profile = pipeline.start(config)
            device = profile.get_device()

            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            (
                model_name,
                coeffs_rs,
                dist_cv,
                pnp_distortion_handling,
            ) = realsense_coeffs_to_opencv(intr)

            try:
                fps = int(round(float(color_profile.fps())))
            except Exception:
                fps = int(self.args.color_fps)

            self.intrinsics = CameraIntrinsics(
                width=int(intr.width),
                height=int(intr.height),
                fps=fps,
                fx=float(intr.fx),
                fy=float(intr.fy),
                cx=float(intr.ppx),
                cy=float(intr.ppy),
                model=model_name,
                coeffs_rs=coeffs_rs,
                dist_coeffs_cv=dist_cv,
                pnp_distortion_handling=pnp_distortion_handling,
                rs_intrinsics=intr,
            )
            self.pose_estimator = CharucoPoseEstimator(self.args, self.intrinsics)
            print(
                "[INFO] RealSense distortion handling: "
                f"model={model_name}, mode={pnp_distortion_handling}, "
                f"coeffs={coeffs_rs.tolist()}"
            )

            self.device_name = _rs_info(device, rs.camera_info.name)
            self.serial_number = _rs_info(device, rs.camera_info.serial_number)
            self.firmware_version = _rs_info(device, rs.camera_info.firmware_version)

            self.pipeline = pipeline
            self.pipeline_profile = profile
            self.connected = True

            # Let auto-exposure / white balance settle before calibration capture.
            for _ in range(int(self.args.warmup_frames)):
                try:
                    pipeline.wait_for_frames(int(self.args.capture_timeout_ms))
                except Exception:
                    break

            self.update_camera_label()
            print("[INFO] RealSense COLOR intrinsics K:")
            print(self.intrinsics.K)
            print(
                "[INFO] RealSense distortion "
                f"model={self.intrinsics.model}, coeffs={self.intrinsics.coeffs_rs}"
            )
            print(
                "[INFO] ChArUco geometric center in native board frame [m]: "
                f"{self.pose_estimator.center_board_native_m}"
            )
            self.set_status("RealSense connected and active COLOR intrinsics loaded.")
            return True
        except Exception as exc:
            traceback.print_exc()
            try:
                pipeline.stop()  # type: ignore[name-defined]
            except Exception:
                pass
            self.pipeline = None
            self.pipeline_profile = None
            self.connected = False
            self.intrinsics = None
            self.pose_estimator = None
            self.update_camera_label()
            self.set_status(f"RealSense connection failed: {exc}")
            return False

    def disconnect_camera(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as exc:
                print(f"[WARN] RealSense pipeline.stop() failed: {exc}")
        self.pipeline = None
        self.pipeline_profile = None
        self.connected = False
        self.intrinsics = None
        self.pose_estimator = None
        self.update_camera_label()

    def reconnect_camera(self) -> None:
        if self.busy:
            self.set_status("Reconnect unavailable during capture/calibration.")
            return
        self.disconnect_camera()
        if self.connect_camera():
            self.set_status("RealSense reconnected. Ready to capture.")

    def capture_color_bgr(self) -> np.ndarray:
        if not self.connected and not self.connect_camera():
            raise RuntimeError("RealSense is not connected.")
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline is unavailable.")

        frames = self.pipeline.wait_for_frames(int(self.args.capture_timeout_ms))
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Captured frameset did not contain a COLOR frame.")
        image = np.asanyarray(color_frame.get_data())
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Unexpected COLOR image shape: {image.shape}")
        return np.ascontiguousarray(image, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Capture + calibration workflow
    # ------------------------------------------------------------------
    def capture_and_calibrate(self) -> None:
        if self.busy:
            self.set_status("Capture/calibration is already in progress.")
            return

        self.busy = True
        try:
            self.update_detection_label(None)
            self.update_pose_label(None)

            if not self.connect_camera():
                return
            assert self.pose_estimator is not None
            assert self.intrinsics is not None

            self.set_status("Capturing RealSense COLOR frame...")
            raw_bgr = self.capture_color_bgr()

            self.set_status("Detecting ChArUco board and estimating pose...")
            detection = self.pose_estimator.detect_and_estimate(raw_bgr)
            self.update_detection_label(detection)

            rmse_text = (
                f"{detection.reprojection_rmse_px:.3f}px"
                if detection.reprojection_rmse_px is not None
                else "n/a"
            )
            total_markers = self.pose_estimator.total_marker_count
            total_corners = self.pose_estimator.total_charuco_corner_count
            marker_text = CharucoPoseEstimator._count_with_ratio(
                detection.marker_count, total_markers
            )
            corner_text = CharucoPoseEstimator._count_with_ratio(
                detection.charuco_corner_count, total_corners
            )
            inlier_ratio = (
                100.0 * detection.pnp_inlier_count / detection.charuco_corner_count
                if detection.charuco_corner_count > 0
                else 0.0
            )
            print(
                "[Piper Camera Calibration] Detection status: "
                f"ArUco markers={marker_text}, "
                f"ChArUco corners={corner_text}, "
                f"PnP inliers={detection.pnp_inlier_count}/"
                f"{detection.charuco_corner_count} ({inlier_ratio:.1f}%), "
                f"RMSE={rmse_text}"
            )

            # Always show detection visualization, including rejected captures.
            preview_label = (
                f"{detection.message} | "
                f"ArUco markers={marker_text} | "
                f"ChArUco corners={corner_text}"
            )
            self._update_preview(detection.annotated_bgr, preview_label)

            if not detection.success:
                self.set_status(detection.message + " No calibration file was overwritten.")
                return

            self.set_status("Transforming RGB optical pose into Piper World / USD camera axes...")
            try:
                calibration = construct_calibration(
                    self.args,
                    detection,
                    self.pose_estimator,
                    self.intrinsics,
                    self.device_name,
                    self.serial_number,
                    self.firmware_version,
                )
            except Exception as exc:
                # Detection may be numerically good yet fail physical pose sanity.
                rejected = detection.annotated_bgr.copy()
                CharucoPoseEstimator._draw_status_text(
                    rejected,
                    [
                        "REJECTED: world-pose sanity check failed",
                        (
                            f"ArUco markers={marker_text}, "
                            f"ChArUco corners={corner_text}, "
                            f"PnP inliers={detection.pnp_inlier_count}/"
                            f"{detection.charuco_corner_count} ({inlier_ratio:.1f}%)"
                        ),
                        str(exc),
                    ],
                    accepted=False,
                )
                self._update_preview(
                    rejected,
                    (
                        f"REJECTED: {exc} | "
                        f"ArUco markers={marker_text} | "
                        f"ChArUco corners={corner_text}"
                    ),
                )
                self.set_status(f"Calibration rejected by world-pose sanity check: {exc}")
                return

            annotated = annotate_world_pose(detection.annotated_bgr, calibration)
            self._update_preview(annotated, "ACCEPTED: camera pose calibrated and saved")

            self.set_status("Saving canonical camera config and calibration evidence...")
            canonical, log_yaml, raw_path, preview_path = save_calibration_files(
                calibration,
                raw_bgr,
                annotated,
                self.args,
            )

            self.last_calibration = calibration
            self.last_saved_config = canonical
            self.update_pose_label(calibration)
            if self.file_label is not None:
                self.file_label.text = f"Saved canonical: {canonical}"

            p = calibration.T_world_isaac_camera_m[:3, 3]
            q = calibration.isaac_quaternion_wxyz
            print("\n[INFO] Camera calibration ACCEPTED")
            print(f"       canonical config: {canonical}")
            print(f"       timestamped YAML: {log_yaml}")
            print(f"       raw RGB:          {raw_path}")
            print(f"       annotated RGB:    {preview_path}")
            print(f"       RMSE:             {calibration.reprojection_rmse_px:.4f} px")
            print(f"       USD camera xyz:   {p}")
            print(f"       USD camera q_wxyz:{q}")
            print("       T_world_color_optical_cv:")
            print(calibration.T_world_color_optical_m)
            print("       T_world_isaac_camera_usd:")
            print(calibration.T_world_isaac_camera_m)
            print()

            self.set_status(
                f"ACCEPTED and saved. RMSE={calibration.reprojection_rmse_px:.3f}px | "
                f"{canonical}"
            )
        except Exception as exc:
            traceback.print_exc()
            self.set_status(f"Capture/calibration error: {exc}")
        finally:
            self.busy = False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            if getattr(self, "keyboard_sub_id", None) is not None:
                self.input_interface.unsubscribe_to_keyboard_events(
                    self.keyboard, self.keyboard_sub_id
                )
                self.keyboard_sub_id = None
        except Exception as exc:
            print(f"[WARN] Keyboard unsubscribe error: {exc}")

        self.disconnect_camera()

        if self.window is not None:
            self.window.visible = False
            self.window = None
        if self.preview_window is not None:
            self.preview_window.visible = False
            self.preview_window = None

        if self._preview_upload_task is not None:
            try:
                self._preview_upload_task.cancel()
            except Exception:
                pass
            self._preview_upload_task = None

        if self.preview_provider is not None:
            try:
                destroy_fn = getattr(self.preview_provider, "destroy", None)
                if callable(destroy_fn):
                    destroy_fn()
            except Exception:
                pass
            self.preview_provider = None

        if self._opencv_preview_fallback:
            try:
                cv2.destroyWindow("RealSense RGB / ChArUco Detection")
            except cv2.error:
                pass

        print("[INFO] Piper RealSense ChArUco calibration controller stopped.")


# -----------------------------------------------------------------------------
# Script Editor re-run safety and application loop
# -----------------------------------------------------------------------------
_CONTROLLER_GLOBAL_NAME = "PIPER_REALSENSE_CHARUCO_CALIBRATION_CONTROLLER"
_previous_controller = globals().get(_CONTROLLER_GLOBAL_NAME)
if _previous_controller is not None:
    try:
        _previous_controller.shutdown()
    except Exception:
        traceback.print_exc()

controller = RealSenseCharucoCalibrationController(ARGS)
globals()[_CONTROLLER_GLOBAL_NAME] = controller

print("\n============================================================")
print("Piper RealSense ChArUco camera calibration is ready.")
print("  Capture + Calibrate button (or C): capture, detect, validate, save")
print(f"  Project root:     {PROJECT_ROOT}")
print(f"  Canonical config: {CANONICAL_CONFIG_PATH}")
print(f"  Calibration logs:{LOG_DIR}")
print("  Board center W:   [0.345, 0.000, 0.000] m")
print("  Board axes:       +X->-X_W, +Y->-Y_W, +Z->+Z_W")
print("============================================================\n")

if SIMULATION_APP is not None:
    try:
        while SIMULATION_APP.is_running():
            SIMULATION_APP.update()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        SIMULATION_APP.close()
