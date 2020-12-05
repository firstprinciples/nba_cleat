import setuptools
import os
import fps_nba_cleat

# Get the version of this package
version = fps_nba_cleat.version

# Get the long description of this package
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fps_nba_cleat',
    version=version,
    author="First Principles",
    author_email="rcpattison@gmail.com",
    description="Colla with cleat street to bet NBA lines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/firstprinciples/projects/nba_cleat",
    packages=setuptools.find_packages(exclude=['unit_tests']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'xlrd',
        'tensorflow'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
