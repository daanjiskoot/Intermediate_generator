from setuptools import setup, find_packages

setup(
    name="intermediate_generator",
    version="0.0.3",
    author="Daan Jiskoot",
    author_email="djiskoot@vuw.leidenuniv.nl",
    description="A molecular intermediate generator for relative binding free energy calculations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/daanjiskoot/Intermediate_generator",
    packages=find_packages(where='generator'),
    classifiers=[
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "openbabel>=3.1",
        "rdkit==2022.09.5",
        "selfies==2.1.1"
    ]
        entry_points={
        'console_scripts': [
            'intermediates = your_module:main'
        ]
    }
)


