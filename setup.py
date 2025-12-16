from setuptools import setup, find_packages

setup(
    name="macecvs",
    version="0.1",
    description="A Python package for MACE-based machine learning collective variables",
    author="Thomas Pigeon",
    author_email="thomas.pigeon@ifpen.fr",
    packages=find_packages(),  # Automatically discover sub-packages
    install_requires=[
        "numpy",                  # For numerical computations
        "torch",                  # PyTorch for machine learning
        "matplotlib",             # For plotting and visualization
        "scipy",                  # Scientific computation
        "ase",                    # Atomic Simulation Environment
        "e3nn",                   # Equivariant neural networks
        "mace-torch"              # MACE MLIP
    ],
    python_requires=">=3.11",      # Specify the minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="MACE, machine learning, ASE, atomic simulations, PyTorch",
    license="MIT",
    url="https://github.com/your-repo/my_package",  # Replace with the actual URL
)

