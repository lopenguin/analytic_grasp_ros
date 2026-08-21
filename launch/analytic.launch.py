from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        ####
        # Publish all robot tfs in URDF (TODO: needed if you don't want to rely on rviz)
        ####
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('agx_arm_description'), 'launch', 'publish.launch.py']),
            launch_arguments={'namespace': 'nero_right',
                              'arm_type': 'nero',
                              'follow': 'true',
                            #   'kinematics_config':'false',
                              }.items()
        ),
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
            parameters=[{"transforms": "/home/agilex/lorenzo/calibration/save"}]
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