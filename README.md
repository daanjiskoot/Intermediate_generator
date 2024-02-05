# Intermediate generator
This python package is designed to create molecular intermediates from two moleculair endpoints. The molecular intermediates are specifically designed for their application in relative binding free energy calculations.

This generation of intermediates works in 2 rounds. In the first generative round the chemical space between two molecular enpoint molecules is populated. After selection of a candidate intermediate that shows high similarity towards both endpoints and complies with certain rules, this molecule is used as the input for the second generative round. During the second process, mutations are performed with the input molecule. From this set, the best intermediate is selected based on the similarity and the LOMAP score towards both original endpoints. 

<img width="449" alt="image" src="https://github.com/daanjiskoot/Intermediate_generator/assets/99884943/822f0603-0cf4-43a6-ad7b-f790591c2f21">

# Prerequisites

- Python (tested with version 3.10)
- Lomap
- OpenEye toolkit (a license is only required for 3D scoring)
- Openbabel
- Ipython

# Installation 

The package can be installed as a PyPI package using:

```pip install intermediate-generator```

OR

Create the shipped conda environment using:

```conda env create -f environment.yml```

# Usage

The minimal command to run the intermediate generator requires to call intermediates with the path to two sdf files with the endpoint molecules and a base directory to save the ouput files:

```intermediates -i1 path/to/ligand1.sdf path/to/ligand2.sdf -b path/to/basedir```

Various more parameters can be set, which can be shown with: 

```intermediates --help``` 

or can be found in the cli.py file.

# Acknowledgements
