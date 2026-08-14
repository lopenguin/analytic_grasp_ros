# Analytic Grasping with Nero Arms

## Visualize the arm in rviz
First, launch the arm state publisher (mode=1 to only publish)
```
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=1 auto_enable:=true
```

Second, launch rviz and map it to the urdf:
```
ros2 launch agx_arm_description display.launch.py namespace:=nero_right
```

The default settings connect to the right gripper; change this to `nero_left` for the left gripper. See the launch file for more config options.

## Nodes
Point cloud publisher:
```
ros2 run analytic_grasp_ros pcd_publisher --ros-args -p pcd_file:=/home/agilex/lorenzo/analytic/pickmeup.pcd
```
Publishes a saved point cloud in the camera frame!

Static tf publisher:
```
ros2 run analytic_grasp_ros calibration_publisher
```
Publishes static transforms from calibration.

Grasp selector:
```
ros2 run analytic_grasp_ros grasp_selection
```
Reads a point cloud message and computes analytic grasps. Publishes them to the tf tree.

## Gripper visualization
To add in gripper visualizations, first run a robot state publisher for the gripper:
```
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(cat /home/agilex/nero_aloha/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/gripper_only.urdf)" \
  -p frame_prefix:=no0/ \
  -r robot_description:=robot2
```
And then a static tf:
```
ros2 run tf2_ros static_transform_publisher \
    0 0 0 0 0 0 \
    grasp_candidate_00 no0/gripper_flange
```

## Launch
First, start the arm controller and camera:
```bash
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=1 auto_enable:=true &
ros2 launch orbbec_camera multi_camera.launch.py \
  front_serial:="CC1WC5200NX" \
  left_serial:="CC1WC5201L8" \
  right_serial:="CC1WC520126"
```
(make sure to type `fg` when closing to kill both of these launches!)

Optionally, launch rviz to view what is going on:
```bash
ros2 launch analytic_grasp_ros view.launch.py
# ros2 launch analytic_grasp_ros view.launch.py config:=view_r.rviz
```

Then, launch all the grasping infrastructure:
TODO: currently this does not work without rviz! (due to some frame weirdness with gripper_palm...)
```bash
ros2 launch analytic_grasp_ros analytic.launch.py
```





TODO:
1. Need a static tf for the gripper grab parts (done)
2. Need some tf logic for the transform: shouldn't be gripper_flange! (done)
3. Make a few launch files! (done)
4. Fix calibration problems

This was very helpful: https://github.com/MetroRobots/rosetta_launch




## TODO
- visualize candidate grasps in rviz (need new node)