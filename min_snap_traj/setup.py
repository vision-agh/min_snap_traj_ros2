from setuptools import find_packages, setup

package_name = "min_snap_traj"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "cvxpy", "numpy"],
    zip_safe=True,
    maintainer="Hubert Szolc",
    maintainer_email="szolc@agh.edu.pl",
    description="Minimum snap trajectory generation for UAVs",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mst_planner = min_snap_traj.mst_planner:main",
        ],
    },
)
