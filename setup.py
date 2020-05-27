import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-nemo",
    version="0.0.7",
    author="Francesco Conti",
    author_email="f.conti@unibo.it",
    description="NEural Minimizer for pytOrch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pulp-platform/nemo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        "torch>=1.3.0",
        "torchvision>=0.4.1",
        "numpy",
        "tqdm",
        "packaging",
        "scikit-learn"
    ]
)
