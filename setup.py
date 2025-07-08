from setuptools import setup, find_packages

setup(
    name="sensing_area_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "opencv-python",
        "albumentations",
        "numpy",
        "scikit-learn",
        "Pillow",
    ],
)
