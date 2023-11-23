from setuptools import setup
# from setuptools import find_packages

package_name = 'webots_drone'

setup(
    name=package_name,
    version='0.1.0',    
    description='Gym wrapper for Webots simulations scene with the DJI Mavic Pro 2 drone',
    url='https://github.com/angel-ayala/gym-webots-drone',
    author='Angel Ayala',
    author_email='aaam@ecomp.poli.br',
    license='GPL-3.0',
    packages=[package_name],
    # packages=find_packages(),
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
