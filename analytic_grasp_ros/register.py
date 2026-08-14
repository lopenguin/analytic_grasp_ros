#!/usr/bin/env python3
"""
Register a live depth point cloud (fixed camera, ROS 2) against a synthetic
view of an articulated URDF, where:

  - The camera guess pose (T_world_camera) is pulled from the TF tree.
  - The joint configuration is pulled from a JointState message
    (e.g. from joint_state_publisher).

Outputs a refined T_world_camera via point-to-plane ICP and prints the residual.
"""

import threading
import numpy as np
import open3d as o3d
import yourdfpy
import xacro

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import JointState
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import TransformException

import os
from ament_index_python.packages import get_package_share_directory

def ros_package_resolver(fname):
    if fname.startswith("package://"):
        pkg, rel = fname[len("package://"):].split("/", 1)
        return os.path.join(get_package_share_directory(pkg), rel)
    if fname.startswith("file://"):
        return fname[len("file://"):]
    return fname

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

def quaternion_to_rotation_matrix(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

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
    return R

import io
import trimesh
# ----------------------------------------------------------------------------
# Model-cloud generation from URDF
# ----------------------------------------------------------------------------
def load_urdf_or_xacro(path, mappings=None):
    """Load a URDF or .xacro file into yourdfpy."""
    if path.endswith(".xacro"):
        # mappings = xacro args, e.g. {"use_gazebo": "false"}
        doc = xacro.process_file(path, mappings=mappings or {})
        urdf_xml = doc.toxml()
        return yourdfpy.URDF.load(io.StringIO(urdf_xml), filename_handler=ros_package_resolver)
    return yourdfpy.URDF.load(path, filename_handler=ros_package_resolver)

def build_model_cloud(urdf_path, joint_cfg, T_world_base, n_points=200_000):
    """
    Load URDF, set joint config, place base at T_world_base (4x4),
    return an Open3D point cloud (world frame) sampled from all link meshes.
    """
    robot = load_urdf_or_xacro(urdf_path)
    # breakpoint()
    t = joint_cfg.pop("gripper", None)
    joint_cfg["gripper_joint1"] = t
    robot.update_cfg(joint_cfg)          # dict {joint_name: angle}
    scene = robot.scene                  # trimesh.Scene, posed in URDF base frame

    combined = o3d.geometry.TriangleMesh()
    #breakpoint()
    for name, geom in scene.geometry.items():
        #breakpoint()
        if isinstance(geom, trimesh.Scene):
            geom = trimesh.util.concatenate([g for g in geom.dump().geometry.values()])

        T_base_geom = scene.graph.get(name)[0]

        m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(geom.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(geom.faces)),
        )
        m.transform(T_base_geom)         # geom -> base
        combined += m

    combined.transform(T_world_base)     # base -> world
    combined.compute_vertex_normals()

    pcd = combined.sample_points_poisson_disk(number_of_points=n_points)
    return pcd


def cull_to_visible(model_pcd, camera_origin_world):
    """Hidden-point removal so the model matches a single-view depth image."""
    diameter = np.linalg.norm(
        np.asarray(model_pcd.get_max_bound()) -
        np.asarray(model_pcd.get_min_bound()))
    radius = diameter * 1000.0
    _, idx = model_pcd.hidden_point_removal(camera_origin_world, radius)
    return model_pcd.select_by_index(idx)


# ----------------------------------------------------------------------------
# Point cloud utilities
# ----------------------------------------------------------------------------
def pc2_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    pts = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z"), skip_nans=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(float))
    return pcd


def preprocess(pcd, voxel):
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pcd


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
class DepthRegistrationNode(Node):
    def __init__(self):
        super().__init__("depth_urdf_registration")

        # -------- parameters --------
        self.declare_parameter("cloud_topic", "/camera_f/depth/points")
        self.declare_parameter("joint_states_topic", "/nero_right/puppet/joint_states")
        self.declare_parameter("urdf_path", "/home/agilex/nero_aloha/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_with_gripper_description.xacro")
        self.declare_parameter("voxel_size", 0.005)          # meters (finest)
        self.declare_parameter("world_frame", "nero_right/world")
        self.declare_parameter("camera_frame", "camera_f_color_optical_frame")
        self.declare_parameter("urdf_base_frame", "nero_right/base_link")

        self.cloud_topic  = self.get_parameter("cloud_topic").value
        self.js_topic     = self.get_parameter("joint_states_topic").value
        self.urdf_path    = self.get_parameter("urdf_path").value
        self.voxel        = self.get_parameter("voxel_size").value
        self.world_frame  = self.get_parameter("world_frame").value
        self.cam_frame    = self.get_parameter("camera_frame").value
        self.base_frame   = self.get_parameter("urdf_base_frame").value

        # -------- shared state --------
        self._lock = threading.Lock()
        self.joint_cfg = None            # {name: position}

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        # -------- I/O --------
        self.create_subscription(
            JointState, self.js_topic, self.joint_cb, 10)
        self.create_subscription(
            PointCloud2, self.cloud_topic, self.cloud_cb, 1)

        self.get_logger().info("Waiting for /joint_states and TF...")

    # ---- pull joint config from joint_state_publisher ----
    def joint_cb(self, msg: JointState):
        with self._lock:
            self.joint_cfg = dict(zip(msg.name, msg.position))

    # ---- pull camera guess (T_world_camera) from the TF tree ----
    def lookup_guess(self, stamp):
        """Return 4x4 T_world_camera from TF, or None if unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.cam_frame,
                rclpy.time.Time())  # latest available
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        T = np.eye(4)
        T[:3,:3] = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def lookup_base(self):
        """Return 4x4 T_world_base (known URDF base pose) from TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f"Base TF lookup failed: {ex}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        T = np.eye(4)
        T[:3,:3] = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def cloud_cb(self, msg: PointCloud2):
        with self._lock:
            joint_cfg = None if self.joint_cfg is None else dict(self.joint_cfg)
        if joint_cfg is None:
            self.get_logger().warn("No joint state yet, skipping.")
            return

        T_world_cam_guess = self.lookup_guess(msg.header.stamp)
        T_world_base = self.lookup_base()
        if T_world_cam_guess is None or T_world_base is None:
            return

        # ---- build & cull model cloud for current joints/pose ----
        model = build_model_cloud(self.urdf_path, joint_cfg, T_world_base)
        model = cull_to_visible(model, T_world_cam_guess[:3, 3])

        # ---- observed cloud (camera frame) -> world using the guess ----
        obs = pc2_to_o3d(msg)
        if len(obs.points) < 100:
            self.get_logger().warn("Sparse cloud, skipping.")
            return
        obs.transform(T_world_cam_guess)

        # ---- coarse-to-fine point-to-plane ICP: correction in world frame ----
        # correction @ (guess-transformed obs) ~= model
        voxels = [self.voxel * 4, self.voxel * 2, self.voxel]
        max_iters = [60, 40, 25]
        correction = np.eye(4)
        result = None

        for scale, (v, it) in enumerate(zip(voxels, max_iters)):
            src = preprocess(obs, v)      # observed (already in world)
            tgt = preprocess(model, v)    # model
            result = o3d.pipelines.registration.registration_icp(
                src, tgt, v, correction,
                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=it))
            correction = result.transformation
            self.get_logger().info(
                f"[scale {scale} voxel={v:.4f}] "
                f"fitness={result.fitness:.4f} "
                f"inlier_rmse={result.inlier_rmse:.5f} m "
                f"corr_set={len(result.correspondence_set)}")

        # ---- refined camera pose in world ----
        T_world_cam = correction @ T_world_cam_guess

        self.get_logger().info(
            f"RESIDUAL  final fitness={result.fitness:.4f}  "
            f"inlier_rmse={result.inlier_rmse:.5f} m")
        self._publish_tf(T_world_cam, msg.header.stamp)

    def _publish_tf(self, T, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.cam_frame + "_refined"
        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])
        q = rotation_matrix_to_quaternion(T[:3,:3])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = DepthRegistrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()