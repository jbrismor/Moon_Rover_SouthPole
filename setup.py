from setuptools import setup, find_packages

setup(
    name="Moon_env",
    version="1.0.0",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "matplotlib",
        "scipy",
        "noise",
        "Pillow",
        "torch",
        "pyvista",
        "heapq",
        "rasterio"
    ],
    description="Moon Rover motion RL environment",
    author="Jorge Bris Moreno",
    license="MIT",
)