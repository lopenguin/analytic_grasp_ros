from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        ####
        # Point cloud publisher (point cloud from found-it)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='pcd_publisher',
            name='pcd_publisher',
            parameters=[{'pcd_file': '/home/agilex/lorenzo/analytic/pickmeup.pcd'}]
        ),
        ####
        # Calibration publisher (static tfs from calibration)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='calibration_publisher',
            name='calibration_publisher',
            parameters=[{"transforms": "/home/agilex/lorenzo/calibration/calibration_2026-08-11_17-55-21.yaml"}]
        ),
        ####
        # Grasp selection (selects and publishes grasps)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='grasp_selection',
            name='grasp_selection',
            parameters=[{"publish_gripper_tfs": True}] # TODO: make this a param?
        )
    ])