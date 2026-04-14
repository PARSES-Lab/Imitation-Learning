from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'object_grasping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'object_grasping', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joeya',
    maintainer_email='joeyallen212@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_grasping_node = object_grasping.object_grasping_node:main',
            'record_joints_node = object_grasping.record_joints:main',
            'replay_joints_node = object_grasping.replay_joints:main',
            'extract_poses_node = object_grasping.extract_poses:main',
            'inference_node = object_grasping.inference:main'
        ],
    },
)
