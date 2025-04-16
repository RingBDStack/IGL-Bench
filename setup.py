from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="IGL_Bench",
    version="0.1.0",
    description="Imbalanced Graph Learning Benchmark",
    url="https://github.com/RingBDStack/IGL-Bench",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,  
    install_requires=[
        "torch>=1.13.1",
        "torch-geometric>=2.1.0",
        "scipy",
        "numpy",
        "dgl",
        "tqdm",
        "scikit_learn",
        "ogb",
        "networkx"
    ],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    keywords=[
        "graph learning",
        "GNN",
        "imbalanced learning",
        "graph neural networks",
        "benchmark"
    ]
)
