from setuptools import setup, find_packages

setup(
    name="nscan",
    version="0.1.0",
    author="Caleb Maresca",
    author_email="marescacc@gmail.com",
    description="News-Stock Cross-Attention Network for multi-stock prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/calebmaresca/nscan",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.34.0",
        "pandas>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)