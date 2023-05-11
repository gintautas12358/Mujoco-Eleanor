from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    new_desired_pose = LaunchConfiguration('new_desired_pose')
    
    # namespace_launch_arg = DeclareLaunchArgument(
    #     'namespace',
    #     default_value='mj_sim_0'
    # )

    name_launch_arg = DeclareLaunchArgument(
        'name',
        default_value='egen'
    )

    namespace_launch_arg = DeclareLaunchArgument(
        'namespace',
        default_value='mj_egen_0'
    )

    # new_saccade_duration_launch_arg = DeclareLaunchArgument(
    #     'new_saccade_duration',
    #     default_value='0.5'
    # )

    new_desired_pose_launch_arg = DeclareLaunchArgument(
        'new_desired_pose',
        default_value='[0.0, 0.4, 0.3, 1.57, 0, 0]'
    )

    set_desired_pose = ExecuteProcess(
        cmd=[[
            'ros2 action send_goal /',
            namespace,
            '/desired_pose_topic ',
            'controller_interface/action/DesiredPose ',
            '"{des_pose: ',
            new_desired_pose,
            '}"'
        ]],
        shell=True
    )

    return LaunchDescription([
        namespace_launch_arg,
        new_desired_pose_launch_arg,
        set_desired_pose,
    ])