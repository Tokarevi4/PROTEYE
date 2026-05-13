from setuptools import setup, find_packages

setup(
    name="proTeye",
    version="0.1.0",
    description="Generative modeling of protein conformational states using geometric deep learning",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "biopython>=1.81",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
)
