#!/usr/bin/env python3
# Calibrate the arm camera relative to the gripper
import sys
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Transform
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data

### HELPERS
def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M

def Tmsg_to_T(msg: TransformStamped):
    t = msg.transform.translation
    q = msg.transform.rotation
    quat = np.array([float(q.x), float(q.y), float(q.z), float(q.w)])
    R = Rot.from_quat(quat)
    return T(R, t)

def T_to_Tmsg(T) -> Transform:
    transform = Transform()
    transform.translation.x = float(T[3,0])
    transform.translation.y = float(T[3,1])
    transform.translation.z = float(T[3,2])
    q = Rot.as_quat(T[:3,:3])
    transform.rotation.x = float(q[0])
    transform.rotation.y = float(q[1])
    transform.rotation.z = float(q[2])
    transform.rotation.w = float(q[3])
    return transform



# Plan:
# 1) Calibrate front camera relative to fixed board and publish transform
#   (here, you can check by comparing the point clouds in rviz)
# 2) Calibrate arm camera relative to fixed board and publish transform
#   (here, you can check by looking at the point clouds in rviz)
# 3) Calibrate arm camera to gripper using hand-eye calibration stuffs
# 4) Save



class CalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration')

        # parameters
        self.declare_parameter("mode", "none") # cal_f, cal_a, cal_handeye

        self.mode = self.get_parameter("mode").value
        self.total_calib = 30

        # tf setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.broadcaster = TransformBroadcaster(self)

        # the transformers we are solving for
        self.T_FcamBoard = None
        self.T_AcamBoard = None
        self.T_AcamGripper = None
        self.K = {}
        self.D = {}

        # image subscriber
        self.sub_fcam_info = self.create_subscription(
            CameraInfo, "camera_f/color/camera_info", self.cb_fcam_info, 1)
        self.sub_acam_info = self.create_subscription(
            CameraInfo, "camera_r/color/camera_info", self.cb_acam_info, 1)
        self.sub_fcam = self.create_subscription(
            CameraInfo, "camera_f/color/image_raw", self.cb_fcam, qos_profile_sensor_data)
        self.sub_acam = self.create_subscription(
            CameraInfo, "camera_r/color/image_raw", self.cb_acam, qos_profile_sensor_data)
        
        # a publish timer
        publish_period_sec = 1.0
        self.tf_pub_timer = self.create_timer(publish_period_sec, self.pub_tfs)

        # TODO: not using this!
        # 1) calibrate front camera relative to fixed board
        # self.T_FcamBoard = self.calibrate_camera("camera_f")

        # # 2) calibrate arm camera relative to fixed board
        # self.T_AcamBoard = self.calibrate_camera("camera_r")

        # # 3) calibrate arm camera to gripper (will require moving the arm)
        # self.T_AcamGripper = self.calibrate_handeye()


    def pub_tfs(self):
        transforms = []

        if self.T_FcamBoard is not None:
            # publish as board -> camera_f_link
            Tmsg_FcamlinkFcam = self.tf_buffer.lookup_transform(
                'camera_f_link',
                'camera_f_color_optical_frame',
                rclpy.time.Time()
            )
            T_FcamlinkFcam = Tmsg_to_T(Tmsg_FcamlinkFcam)
            T_BoardFcamlink = np.linalg.inv(T_FcamlinkFcam @ self.T_FcamBoard)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg
            transform.header.frame_id = "calibration_board"
            transform.child_frame_id = "camera_f_link"
            transform.transform = T_to_Tmsg(T_BoardFcamlink)
            transforms.append(transform)


        if self.T_AcamBoard is not None:
            # publish as board -> camera_r_link
            Tmsg_AcamlinkAcam = self.tf_buffer.lookup_transform(
                'camera_r_link',
                'camera_r_color_optical_frame',
                rclpy.time.Time()
            )
            T_AcamlinkAcam = Tmsg_to_T(Tmsg_AcamlinkAcam)
            T_BoardAcamlink = np.linalg.inv(T_AcamlinkAcam @ self.T_AcamBoard)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg
            transform.header.frame_id = "calibration_board"
            transform.child_frame_id = "camera_r_link"
            transform.transform = T_to_Tmsg(T_BoardAcamlink)
            transforms.append(transform)


        if self.T_AcamGripper is not None:
            # publish as gripper_base -> camera_r_link
            Tmsg_AcamlinkAcam = self.tf_buffer.lookup_transform(
                'camera_r_link',
                'camera_r_color_optical_frame',
                rclpy.time.Time()
            )
            T_AcamlinkAcam = Tmsg_to_T(Tmsg_AcamlinkAcam)
            T_GripperbaseAcamlink = np.linalg.inv(T_AcamlinkAcam @ self.T_AcamGripper)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg
            transform.header.frame_id = "nero_right/gripper_base" # TODO: base or flange?
            transform.child_frame_id = "camera_r_link"
            transform.transform = T_to_Tmsg(T_GripperbaseAcamlink)
            transforms.append(transform)

        # publish!
        if len(transforms) > 0:
            self.broadcaster.sendTransform(transforms)


    ## Camera intrinsics!
    def cb_info(self, cam_name, msg: CameraInfo):
        self.K[cam_name] = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D[cam_name] = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(f"Got {cam_name} intrinsics")
    def cb_fcam_info(self, msg: CameraInfo):
        self.cb_info("cam_f", msg)
        # Tear down the subscription so we never pay for it again.
        self.destroy_subscription(self.sub_fcam_info)
        self.cb_fcam_info = None
    def cb_acam_info(self, msg: CameraInfo):
        self.cb_info("cam_a", msg)
        # Tear down the subscription so we never pay for it again.
        self.destroy_subscription(self.sub_acam_info)
        self.cb_acam_info = None

    ## Camera image subscribers!
    def cb_fcam(self, msg: Image):
        if self.mode != "cal_f":
            return

        

    def cb_acam(self, msg: Image):
        if self.mode == "cal_a":
            pass
        elif self.mode == "cal_handeye":
            # TODO: this may be hard to align with the 
            # forward kinematics
            # probably want to use a message filter (TODO)
            pass

    


    def detect_board(self, cam_name):
        pass

    def get_cam_pose(self, cam_name, objpoints, imgpoints):
        """
        Solve PnP for camera pose relative to board
        """
        all_object_points = np.concatenate(
            [p.reshape(-1, 3) for p in objpoints],
            axis=0,
        ).astype(np.float32)

        all_image_points = np.concatenate(
            [p.reshape(-1, 2) for p in imgpoints],
            axis=0,
        ).astype(np.float32)

        ok, rvec, tvec, err = cv.solvePnPGeneric(
            all_object_points,
            all_image_points,
            camK,
            camdc,
            flags=cv.SOLVEPNP_ITERATIVE #cv.SOLVEPNP_SQPNP
        )
        if not ok:
            raise RuntimeError("Could not estimate the fixed board-camera pose")
        print("Reprojection errors:")
        print(err)

        # more errors (just for fun)
        img_pred, _ = cv.projectPoints(
            all_object_points,
            rvec[0],
            tvec[0],
            camK,
            camdc
        )
        err = img_pred.reshape(-1, 2) - all_image_points.reshape(-1, 2)
        rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        mean_err = np.mean(np.linalg.norm(err, axis=1))
        max_err = np.max(np.linalg.norm(err, axis=1))
        print(f"rmse: {rmse} px")
        print(f"mean_err: {mean_err} px")
        print(f"max_err: {max_err} px")


        R_CamBoard, _ = cv.Rodrigues(rvec[0])

        T_CamBoard = np.eye(4)
        T_CamBoard[:3,:3] = R_CamBoard
        T_CamBoard[:3,3] = tvec[0].flatten()
        return T_CamBoard


    def calibrate_camera(self, cam_name):
        print("-----------------")
        print(f"Calibrate {cam_name}")
        print("Fix the calibration board in view of the camera.")
        input(f"Begin?")

        obj_points = []
        img_points = []
        while True:
            if len(obj_points) > self.total_calib:
                break

            img, pts = self.detect_board(cam_name)
            if img is None:
                continue
            # show images
            cv.imshow("calibration", img)
            cv.waitKey(1)

            if pts is None:
                continue
            # save
            obj_points.append(pts[0])
            img_points.append(pts[1])

        cv.destroyAllWindows()

        # compute pose using pnp
        T_CamBoard = self.get_cam_pose(cam_name, obj_points, img_points)
        return T_CamBoard




    def calibrate_handeye(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CalibrationNode()
        rclpy.spin(node)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())