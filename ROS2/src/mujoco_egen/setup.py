from setuptools import setup
import os
from glob import glob

package_name = 'mujoco_egen'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'mujoco_egen.utils', 'mujoco_egen.mujoco_env'],
    include_package_data=True,
    package_data={'mujoco_egen': ['cfg/cfg.yaml', 'kuka/meshes/*','kuka/*'],},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='palinauskas',
    maintainer_email='palinauskas@fortiss.org',
    description='Generates event-based data',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'impedance_controller_server = mujoco_egen.impedance_controller_server:main',
            'test = mujoco_egen.impedance_controller_server_test:main',
        ],
    },
)
