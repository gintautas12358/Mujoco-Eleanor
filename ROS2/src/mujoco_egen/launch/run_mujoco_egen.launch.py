from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mujoco_egen',
            namespace='mj_egen_0',
            executable='impedance_controller_server',
            name='egen'
        ),
    ])