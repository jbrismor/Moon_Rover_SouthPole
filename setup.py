from setuptools import setup, find_packages

setup(
    name="Moon_Rover",
    version="1.1.1",
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
        "rasterio"
    ],
    description="Moon Rover motion RL environment",
    author="Jorge Bris Moreno",
    license="MIT",
)