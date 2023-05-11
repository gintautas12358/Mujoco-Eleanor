# Esim MuJoCo simulation with ROS2

Uses Galactic Geochelone version with Ubuntu 20.04

## Install ROS2

follow the steps in https://docs.ros.org/en/galactic/Installation.html

## Install colcon (a build tool)

- `sudo apt-get install python3-empy`
- `sudo apt install python3-colcon-common-extensions`
- `pip install catkin_pkg`
- `pip install lark`


## Build packages

All packages:
- `colcon build`

Only specific packages:
- `colcon build --packages-select mujoco_egen`

## Install local esim_torch before running 

- `pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`

- `pip install rpg_vid2e/esim_torch/`

- `pip install rpg_vid2e/upsampling`

- `pip install -r rpg_vid2e/`

- `pip install mujoco==2.1.5`

## Run mujoco_egen node

- `source /opt/ros/galactic/setup.bash`
- `. install/setup.bash `
- `ros2 run mujoco_egen impedance_controller_server`

## Sending goal poses to the action server

Possible goal poses are are defined as ROS2 parameters and can be view with:
- `ros2 param list`

Sending a request to the DesiredPoseName action server example
- `ros2 action send_goal /desired_pose_name_topic controller_interface/action/DesiredPoseName "{des_pose_name: "LOOK"}"`
- `ros2 action send_goal --feedback /desired_pose_name_topic controller_interface/action/DesiredPoseName "{des_pose_name: "LOOK"}"`

Sending a request to the Saccades action server example
- `ros2 action send_goal --feedback /saccades_topic controller_interface/action/Saccades "{duration: 3.0}"`

Sending a request to the Random Saccades action server example
- `ros2 action send_goal --feedback /random_saccades_topic controller_interface/action/Saccades "{duration: 3.0}"`

## Available launch files

For starting mujoco:
- run_mujoco_egen.launch.py

For selecting desired pose name:
- run_set_desired_pose.launch.py

For running saccades:
- run_saccades.launch.py

Complementary to run_saccades.launch.py
- template_saccades.launch.py

Command example:

`ros2 launch mujoco_egen run_mujoco_egen.launch.py`

If running launch files, one needs to becareful to add the correct namespaces and name.

In use are:

- namescape: mj_sin_0
- name: sim

## ROS2 structure

impedance_controller_server_node:
- Subscribers:

- Publishers: \
    /camera_events_topic: camera_event_data_interface/msg/CameraEvents \
    /tf: tf2_msgs/msg/TFMessage

- Service Servers: \
    /impedance_controller_server_node

- Service Clients:

- Action Servers: \
    /desired_pose_name_topic: controller_interface/action/DesiredPoseName

- Action Clients:


