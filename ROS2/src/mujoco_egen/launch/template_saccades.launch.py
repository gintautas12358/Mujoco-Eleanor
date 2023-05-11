from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    name = LaunchConfiguration('name')
    namespace = LaunchConfiguration('namespace')

    name_launch_arg = DeclareLaunchArgument(
        'name',
        default_value='egen'
    )

    namespace_launch_arg = DeclareLaunchArgument(
        'namespace',
        default_value='mj_egen_0'
    )
    
    start_saccading = ExecuteProcess(
        cmd=[[
            'ros2 action send_goal ',
            namespace,
            '/saccades_topic ',
            'controller_interface/action/Saccades2 ',
            '"{}"'
        ]],
        shell=True
    )

    return LaunchDescription([
        name_launch_arg,
        namespace_launch_arg,
        start_saccading      
    ])