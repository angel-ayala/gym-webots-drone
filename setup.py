from setuptools import setup

package_name = 'webots_drone'

data_files = list()
data_files.append(('share/' + package_name + '/worlds',
                   ['worlds/forest_tower.wbt',
                    'worlds/forest_tower_200x200_simple.wbt',
                    'worlds/crazyflie_2.5x2.5_simple.wbt']))

data_files.append(('share/' + package_name + '/protos',
                   ['protos/RadioController.proto',
                    'protos/FireSmoke.proto']))

setup(
    name=package_name,
    version='2.1.0',
    description='Gym wrapper for Webots simulation scenesa with the DJI Mavic Pro 2 drone',
    url='https://github.com/angel-ayala/gym-webots-drone',
    author='Angel Ayala',
    author_email='aaam@ecomp.poli.br',
    license='GPL-3.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['gym==0.26.0',
                      'simple_pid==2.0.0',
                      'opencv-python==4.8.1.*'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Games/Entertainment :: Simulation',
    ],
)
