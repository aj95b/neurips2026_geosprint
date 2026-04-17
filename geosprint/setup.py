from setuptools import setup, find_packages

setup(
    name="geosprint",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "torch>=2.0",
        "diffusers>=0.25",
        "transformers>=4.30",
        "pytorch-fid>=0.3",
        "matplotlib>=3.5",
        "tqdm>=4.60",
    ],
)
