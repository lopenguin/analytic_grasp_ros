from __future__ import annotations

import threading

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


# ============================================================================
# Keep all constants and grasp-generation functions from the original script:
#
#   CROP_MIN, CROP_MAX, P_GS_G, ...
#   make_transform
#   invert_transform
#   transform_points
#   rotation_x
#   grasp_candidate_cost
#   generate_antipodal_grasp_candidate
#   sample_grasps
# ============================================================================


# ---------------------------------------------------------------------------
# Gripper geometry (Schunk WSG-50, in gripper frame G). Units: meters.
#
# `CROP_MIN`/`CROP_MAX` describe the closing volume swept between the two
# fingers -- a cloud point inside this box will end up between the fingers when
# the hand closes. These constants are taken verbatim from the reference
# notebook.
# ---------------------------------------------------------------------------
CROP_MIN = np.array([-0.05, 0.1, -0.00625])
CROP_MAX = np.array([0.05, 0.1125, 0.00625])

# Offset, expressed in the gripper frame, from the gripper origin G to the
# sample point S that we align the grasp against (also from the reference).
P_GS_G = np.array([0.054 - 0.01, 0.10625, 0.0])

# Roll angles (about the surface normal) tried for each sample, ordered from
# the center outward, exactly as in the reference notebook.
MIN_ROLL = -np.pi / 3.0
MAX_ROLL = np.pi / 3.0
ROLL_ALPHA = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])

# Analytic collision model of the gripper, as a set of axis-aligned boxes in
# the gripper frame G, each given as (min_corner, max_corner). These roughly
# bound the two fingers and the palm/backplate of the WSG-50 so that we can
# reject grasps whose solid parts would intersect the observed point cloud
# (Drake did this with the true mesh + signed-distance queries).
#
# The fingers sit just outside the closing region along +/-x and extend from
# the palm (small y) out to the fingertips (past the crop box in y). The palm
# spans the gap behind the fingers.
# x: span between fingertips
# y: depth from palm to fingertips
_FINGER_HALF_THICKNESS = 0.012  # half-width of a finger along x
_FINGER_HALF_DEPTH = 0.0125  # half-width of a finger along z
_FINGER_Y_MIN = 0.02  # palm side of the fingers
_FINGER_Y_MAX = 0.092  # fingertips
_FINGER_X_INNER = 0.05  # inner face of finger == edge of closing region
_PALM_HALF_WIDTH = 0.08  # palm extent along x
_PALM_HALF_DEPTH = 0.021  # palm extent along z
_PALM_Y_MIN = 0.0
_PALM_Y_MAX = 0.02

GRIPPER_COLLISION_BOXES = [
    # left finger (-x side)
    (
        np.array([-_FINGER_X_INNER - 2 * _FINGER_HALF_THICKNESS, _FINGER_Y_MIN, -_FINGER_HALF_DEPTH]),
        np.array([-_FINGER_X_INNER, _FINGER_Y_MAX, _FINGER_HALF_DEPTH]),
    ),
    # right finger (+x side)
    (
        np.array([_FINGER_X_INNER, _FINGER_Y_MIN, -_FINGER_HALF_DEPTH]),
        np.array([_FINGER_X_INNER + 2 * _FINGER_HALF_THICKNESS, _FINGER_Y_MAX, _FINGER_HALF_DEPTH]),
    ),
    # palm / backplate spanning the gap behind the fingers
    (
        np.array([-_PALM_HALF_WIDTH, _PALM_Y_MIN, -_PALM_HALF_DEPTH]),
        np.array([_PALM_HALF_WIDTH, _PALM_Y_MAX, _PALM_HALF_DEPTH]),
    ),
]


# ---------------------------------------------------------------------------
# Small rigid-transform helpers (4x4 homogeneous matrices), so we don't need
# Drake's RigidTransform / RotationMatrix.
# ---------------------------------------------------------------------------
def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p
    return X


def invert_transform(X: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R = X[:3, :3]
    p = X[:3, 3]
    Xinv = np.eye(4)
    Xinv[:3, :3] = R.T
    Xinv[:3, 3] = -R.T @ p
    return Xinv


def transform_points(X: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply X to an (N, 3) array of points, returning (N, 3)."""
    return pts @ X[:3, :3].T + X[:3, 3]


def rotation_x(theta: float) -> np.ndarray:
    """Rotation matrix about the x axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------------------
# Grasp cost (port of GraspCandidateCost)
# ---------------------------------------------------------------------------
def grasp_candidate_cost(
    X_G: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    adjust_X_G: bool = False,
    verbose: bool = False,
):
    """Score a grasp pose against the point cloud.

    Lower is better. Returns (cost, X_G) where X_G may have been recentered
    along the finger-closing axis when `adjust_X_G` is True (mirrors the
    reference behavior). An infinite cost means the candidate is infeasible
    (nothing to grasp between the fingers, or a collision).
    """
    X_GW = invert_transform(X_G)
    p_GC = transform_points(X_GW, points)  # cloud in gripper frame, (N, 3)

    # Points that fall inside the closing region between the fingers.
    inside = np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)
    n_inside = int(np.sum(inside))

    if n_inside == 0:
        if verbose:
            print("cost: inf  (no points between the fingers)")
        return np.inf, X_G

    # Optionally recenter the gripper along its x axis so the grasped points are
    # centered between the fingers.
    if adjust_X_G:
        p_GC_x = p_GC[inside, 0]
        center_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        shift_G = np.array([center_x, 0.0, 0.0])
        new_p = X_G[:3, :3] @ shift_G + X_G[:3, 3]
        X_G = X_G.copy()
        X_G[:3, 3] = new_p
        X_GW = invert_transform(X_G)
        p_GC = transform_points(X_GW, points)
        inside = np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)

    # Collision check: any cloud point inside a solid part of the gripper makes
    # this grasp infeasible (Drake used signed-distance to the WSG mesh).
    for lo, hi in GRIPPER_COLLISION_BOXES:
        if np.any(np.all((p_GC >= lo) & (p_GC <= hi), axis=1)):
            if verbose:
                print("cost: inf  (gripper collides with the point cloud)")
            return np.inf, X_G

    # Ground plane collision check: any solid part of the gripper goes below
    # the lowest part of the object (with some tolerance)
    # TODO: this part is broken!
    # tol = 0.05
    # p_minz_W = points[np.argmin(points[:,2]), 2] - tol
    # print(p_minz_W)
    # for lo, hi in GRIPPER_COLLISION_BOXES:
    #     lo_W = transform_points(X_G, lo)
    #     hi_W = transform_points(X_G, hi)
    #     if lo_W[2] < p_minz_W or hi_W[2] < p_minz_W:
    #         print(lo_W)
    #         print(hi_W)
    #         print(p_minz_W)
    #         # input()
    #         if verbose:
    #             print("cost: inf  (gripper collides with the ground)")
    #         return np.inf, X_G

    # Normals of the grasped points, expressed in the gripper frame.
    n_GC = normals[inside] @ X_GW[:3, :3].T  # (n_inside, 3)

    # Penalize deviation of the gripper from a straight-down approach.
    # In the reference this is 20 * R_G[2, 1]: the world-z component of the
    # gripper's +y (approach) axis. For a top-down grasp the approach axis
    # points toward world -z, giving R_G[2,1] = -1 and a reward of -20.
    cost = 20.0 * X_G[2, 1]

    # Reward antipodal alignment: surface normals aligned with the finger
    # closing axis (gripper x). Sum of squared dot products.
    cost -= float(np.sum(n_GC[:, 0] ** 2))

    if verbose:
        print(f"cost: {cost:.4f}  (points grasped: {n_inside})")
    return cost, X_G


# ---------------------------------------------------------------------------
# Candidate generation (port of GenerateAntipodalGraspCandidate)
# ---------------------------------------------------------------------------
def generate_antipodal_grasp_candidate(points, normals, rng):
    """Pick a random surface point and try to align a top-down antipodal grasp.

    Returns (cost, X_G). cost is inf (and X_G is None) if no feasible roll was
    found for the sampled point.
    """
    index = rng.integers(0, points.shape[0])
    p_WS = points[index]
    n_WS = normals[index]

    norm = np.linalg.norm(n_WS)
    if not np.isclose(norm, 1.0):
        n_WS = n_WS / norm

    # Gripper x axis aligns with the surface normal (the finger-closing axis
    # presses along the normal for an antipodal grasp).
    Gx = n_WS
    # Make an orthonormal y axis aligned as much as possible with world down.
    down = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(down, Gx)) > 1.0 - 1e-6:
        # Normal points straight up/down; a from-above grasp is ill-defined.
        return np.inf, None
    Gy = down - np.dot(down, Gx) * Gx
    Gy /= np.linalg.norm(Gy)
    Gz = np.cross(Gx, Gy)
    R_WG = np.column_stack((Gx, Gy, Gz))

    # Try rolls about the normal, from the center out; keep the first feasible.
    for theta in MIN_ROLL + (MAX_ROLL - MIN_ROLL) * ROLL_ALPHA:
        R_WG2 = R_WG @ rotation_x(theta)
        # Place G so that the sample point S lands at P_GS_G in the gripper.
        p_WG = p_WS - R_WG2 @ P_GS_G
        X_G = make_transform(R_WG2, p_WG)
        cost, X_G = grasp_candidate_cost(X_G, points, normals, adjust_X_G=True)
        if np.isfinite(cost):
            return cost, X_G

    return np.inf, None


def sample_grasps(points, normals, num_candidates=200, seed=None):
    """Sample many candidates and return them sorted by cost (best first)."""
    rng = np.random.default_rng(seed)
    costs, poses = [], []
    for _ in range(num_candidates):
        cost, X_G = generate_antipodal_grasp_candidate(points, normals, rng)
        if np.isfinite(cost):
            costs.append(cost)
            poses.append(X_G)
    order = np.argsort(costs)
    return [(costs[i], poses[i]) for i in order]



# ----------------------------------------------------------------
# ROS utils

def transform_msg_to_matrix(msg: TransformStamped) -> np.ndarray:
    """Convert a ROS TransformStamped into a 4x4 homogeneous transform."""
    t = msg.transform.translation
    q = msg.transform.rotation

    # ROS quaternion order is x, y, z, w.
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Received TF transform with a zero-length quaternion.")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    R = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )

    p = np.array([t.x, t.y, t.z], dtype=np.float64)
    return make_transform(R, p)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized ROS-order [x, y, z, w]."""
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Read finite XYZ points from a ROS PointCloud2 message."""
    raw_points = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )

    # Newer ROS 2 releases return a structured NumPy array.
    if isinstance(raw_points, np.ndarray) and raw_points.dtype.names is not None:
        xyz = np.stack(
            (
                np.asarray(raw_points["x"]).reshape(-1),
                np.asarray(raw_points["y"]).reshape(-1),
                np.asarray(raw_points["z"]).reshape(-1),
            ),
            axis=1,
        ).astype(np.float64)
    else:
        # Compatible with ROS 2 releases where read_points returns an iterator.
        xyz = np.asarray(list(raw_points), dtype=np.float64)

        if xyz.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        xyz = xyz.reshape((-1, 3))

    return xyz[np.all(np.isfinite(xyz), axis=1)]


def preprocess_xyz_cloud(
    points_B: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """Downsample a base-frame cloud and estimate outward-facing normals."""
    if points_B.shape[0] < 4:
        raise ValueError("Point cloud has fewer than four valid points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_B)

    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    points = np.asarray(pcd.points)

    if points.shape[0] < 4:
        raise ValueError(
            "Too few points remain after voxel downsampling; reduce voxel_size."
        )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=min(30, points.shape[0]),
        )
    )

    # Match the normal-orientation behavior in the original script.
    pcd.orient_normals_consistent_tangent_plane(
        k=min(30, points.shape[0] - 1)
    )

    normals = np.asarray(pcd.normals)
    centroid = points.mean(axis=0)

    outward = points - centroid
    needs_flip = np.sum(normals * outward, axis=1) < 0.0
    normals[needs_flip] *= -1.0

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return points, normals, pcd


class GraspSelection(Node):
    def __init__(self):
        super().__init__("grasp_selection")

        # --------------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------------
        self.declare_parameter("pcd_topic", "pointcloud")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("candidates", 200)
        self.declare_parameter("top_k", 5)
        self.declare_parameter("voxel_size", 0.005)
        self.declare_parameter("seed", 0)
        self.declare_parameter("namespace", "nero_right")
        self.declare_parameter("process_once", True)
        self.declare_parameter("publish_period", 0.25)
        self.declare_parameter("publish_gripper_tfs", False)
        self.declare_parameter("execute_number", -1)

        # Optional: publishes each candidate as PoseStamped too.
        # TF itself remains published on /tf.
        self.declare_parameter("publish_pose_topics", False)

        self.pcd_topic = str(self.get_parameter("pcd_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value).lstrip("/")
        self.candidates = int(self.get_parameter("candidates").value)
        self.top_k = int(self.get_parameter("top_k").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.seed = int(self.get_parameter("seed").value)
        self.namespace = str(self.get_parameter("namespace").value).strip("/")
        self.process_once = bool(self.get_parameter("process_once").value)
        self.publish_period = max(
            0.01,
            float(self.get_parameter("publish_period").value),
        )
        self.publish_gripper_tfs = bool(self.get_parameter("publish_gripper_tfs").value)
        self.publish_pose_topics = bool(
            self.get_parameter("publish_pose_topics").value
        )

        if not self.namespace:
            self.namespace = ""
        else:
            # add namespace to base_frame
            self.base_frame = f"{self.namespace}/{self.base_frame}"

        # --------------------------------------------------------------------
        # Internal state
        # --------------------------------------------------------------------
        self._lock = threading.Lock()
        self._processing_cloud = False
        self._processed_once = False

        self.points_B: np.ndarray | None = None
        self.normals_B: np.ndarray | None = None
        self.pcd: o3d.geometry.PointCloud | None = None
        self.ranked_grasps: list[tuple[float, np.ndarray]] = []

        # TF listener: looks up T_base_cloud.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------------------------
        self.setup_publishers()
        self.setup_subscribers()

        self.get_logger().info(
            f"Waiting for PointCloud2 messages on '{self.pcd_topic}'."
        )

    def setup_publishers(self):
        """Create the TF broadcaster and optional per-grasp PoseStamped topics."""
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_publishers = []
        if self.publish_pose_topics:
            for rank in range(self.top_k):
                topic = f"/grasp_candidate_{rank:02d}/pose"
                publisher = self.create_publisher(PoseStamped, topic, 10)
                self.pose_publishers.append(publisher)

        self.publish_timer = self.create_timer(
            self.publish_period,
            self.publish_grasps,
        )

        if self.publish_gripper_tfs:
            self.publisher_gripper_timer = self.create_timer(
                self.publish_period,
                self.publish_static_gripper_tfs
            )

            self.pub_move = self.create_publisher(PoseStamped, "nero_right/control/move_p", 1)

    def setup_subscribers(self):
        """Subscribe to the sensor point-cloud stream."""
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            self.pcd_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

    def pointcloud_callback(self, msg: PointCloud2):
        """Process the first usable point cloud, or every cloud if requested."""
        with self._lock:
            if self._processing_cloud:
                return

            if self.process_once and self._processed_once:
                return

            self._processing_cloud = True

        success = False

        try:
            points_B, normals_B, pcd = self.load_point_cloud(msg)

            with self._lock:
                self.points_B = points_B
                self.normals_B = normals_B
                self.pcd = pcd

            self.sample_grasps()

            success = True

        except TransformException as exc:
            # This often occurs briefly at startup, before TF is populated.
            self.get_logger().debug(
                f"Cannot transform cloud from '{msg.header.frame_id}' "
                f"to '{self.base_frame}' yet: {exc}"
            )

        except Exception as exc:
            self.get_logger().error(f"Could not process point cloud: {exc}")

        finally:
            with self._lock:
                self._processing_cloud = False
                if success:
                    self._processed_once = True

    def load_point_cloud(
        self,
        msg: PointCloud2,
    ) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
        """
        Read PointCloud2, transform its points into base_link, downsample,
        and estimate normals.

        Returned pose convention:
            points_B: points expressed in base_frame.
            Therefore each grasp X_G returned by sample_grasps is T_base_gripper.
        """
        cloud_frame = msg.header.frame_id.lstrip("/")

        if not cloud_frame:
            raise ValueError("Incoming PointCloud2 has an empty header.frame_id.")

        points_C = pointcloud2_to_xyz(msg)
        if points_C.shape[0] == 0:
            raise ValueError("Incoming PointCloud2 has no finite XYZ points.")

        # lookup_transform(target, source, time) returns T_target_source.
        if cloud_frame == self.base_frame:
            X_BC = np.eye(4)
        else:
            cloud_stamp = Time.from_msg(msg.header.stamp)

            transform_BC = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_frame,
                cloud_stamp,
            )
            X_BC = transform_msg_to_matrix(transform_BC)

        # Transform raw cloud points before normal estimation, so all subsequent
        # grasp computation is consistently expressed in base_link.
        points_B_raw = transform_points(X_BC, points_C)

        points_B, normals_B, pcd = preprocess_xyz_cloud(
            points_B_raw,
            voxel_size=self.voxel_size,
        )

        self.get_logger().info(
            f"Loaded cloud: {points_C.shape[0]} raw points, "
            f"{points_B.shape[0]} points after downsampling."
        )

        return points_B, normals_B, pcd

    def sample_grasps(self):
        """Run the original antipodal sampler on the most recent base-frame cloud."""
        with self._lock:
            if self.points_B is None or self.normals_B is None:
                raise RuntimeError("Cannot sample grasps before loading a cloud.")

            points_B = self.points_B.copy()
            normals_B = self.normals_B.copy()

        ranked = sample_grasps(
            points_B,
            normals_B,
            num_candidates=self.candidates,
            seed=self.seed,
        )

        ranked = ranked[: self.top_k]

        with self._lock:
            self.ranked_grasps = ranked

        if ranked:
            self.get_logger().info(
                f"Found {len(ranked)} feasible grasp candidates."
            )

            for rank, (cost, X_BG) in enumerate(ranked):
                p = X_BG[:3, 3]
                self.get_logger().info(
                    f"  #{rank:02d}: cost={cost:.4f}, "
                    f"position=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                )
        else:
            self.get_logger().warn(
                "No feasible grasps found. Try more candidates or a smaller voxel size."
            )

    def grasp_frame_name(self, rank: int) -> str:
        """Return a valid child frame name for one ranked candidate."""
        return f"/grasp_candidate_{rank:02d}"

    def pose_from_matrix(
        self,
        X_BG: np.ndarray,
        stamp,
    ) -> PoseStamped:
        """Build a PoseStamped equivalent of T_base_gripper."""
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.base_frame

        qx, qy, qz, qw = rotation_matrix_to_quaternion(X_BG[:3, :3])

        pose.pose.position.x = float(X_BG[0, 3])
        pose.pose.position.y = float(X_BG[1, 3])
        pose.pose.position.z = float(X_BG[2, 3])

        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        return pose

    def publish_grasps(self):
        """
        Publish the current top-k grasps as TF frames:

          base_link
            ├── camera_r/grasp_candidate_00
            ├── camera_r/grasp_candidate_01
            └── ...

        All TF transforms are sent through the normal /tf transport.
        """
        with self._lock:
            ranked = [
                (cost, X_BG.copy())
                for cost, X_BG in self.ranked_grasps
            ]

        if not ranked:
            return

        stamp = self.get_clock().now().to_msg()
        transforms = []

        for rank, (_, X_BG) in enumerate(ranked):
            transform = TransformStamped()

            transform.header.stamp = stamp
            transform.header.frame_id = self.base_frame
            transform.child_frame_id = self.grasp_frame_name(rank)

            qx, qy, qz, qw = rotation_matrix_to_quaternion(X_BG[:3, :3])

            transform.transform.translation.x = float(X_BG[0, 3])
            transform.transform.translation.y = float(X_BG[1, 3])
            transform.transform.translation.z = float(X_BG[2, 3])

            transform.transform.rotation.x = float(qx)
            transform.transform.rotation.y = float(qy)
            transform.transform.rotation.z = float(qz)
            transform.transform.rotation.w = float(qw)

            transforms.append(transform)

            # Optional ordinary ROS topics for consumers that do not use TF.
            if self.publish_pose_topics and rank < len(self.pose_publishers):
                self.pose_publishers[rank].publish(
                    self.pose_from_matrix(X_BG, stamp)
                )

        self.tf_broadcaster.sendTransform(transforms)

        # optionally execute the pose!
        publish_number = self.get_parameter('execute_number').get_parameter_value().integer_value
        if publish_number >= 0:
            X_BG = ranked[publish_number][1]
            # TODO: this seems to publish in the wrong frame!
            self.pub_move.publish(self.pose_from_matrix(ranked[publish_number][1], stamp))

    def publish_static_gripper_tfs(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []
        X_GP = np.eye(4)
        # map x->y, y->z, z->x
        X_GP[:3,:3] = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]]).T
        
        # lookup transform from palm to flange
        transform_BC = self.tf_buffer.lookup_transform(
            f"{self.namespace}/gripper_palm",
            f"{self.namespace}/gripper_flange",
            self.get_clock().now().to_msg(),
        )
        X_PF = transform_msg_to_matrix(transform_BC)
        X_GF = X_GP @ X_PF

        for k in range(self.top_k):
            # robot to gripper
            transform = TransformStamped()

            transform.header.stamp = stamp
            transform.header.frame_id = self.grasp_frame_name(k)
            transform.child_frame_id = f"no{k}/gripper_flange"
            # TODO: should be gripper center or something???

            transform.transform.translation.x = float(X_GF[0, 3])
            transform.transform.translation.y = float(X_GF[1, 3])
            transform.transform.translation.z = float(X_GF[2, 3])

            qx, qy, qz, qw = rotation_matrix_to_quaternion(X_GF[:3, :3])
            transform.transform.rotation.x = float(qx)
            transform.transform.rotation.y = float(qy)
            transform.transform.rotation.z = float(qz)
            transform.transform.rotation.w = float(qw)

            transforms.append(transform)
            
        self.tf_broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = GraspSelection()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        rclpy.logging.get_logger("grasp_selection").error(str(exc))

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()