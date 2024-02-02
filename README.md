# Intermediate generator
This python package is designed to create molecular intermediates, specifically for their application in relative binding free energy calculations.

This generation of intermediates works in 2 rounds. In the first generative round the chemical space between two enpoint molecules is populated. After selection of a candidate intermediate that shows high similarity towards both endpoints and complies with certain rules, this molecule is used as the input for the second generative round. During the second process, a lot of mutations are made to the input molecule. From this set, the best intermediate is selected based on the similarity and the LOMAP score towards both original endpoints. 

<img width="449" alt="image" src="https://github.com/daanjiskoot/Intermediate_generator/assets/99884943/822f0603-0cf4-43a6-ad7b-f790591c2f21">


# Installation of the package

The package can be installed as a PyPI package using:
```pip install intermediate-generator```

additional dependencies can be installed with:

```conda install -c openeye openeye-toolkits```

```other```

```other```

OR

All requirements can be installed by solving the environment through(to do):

```conda env create -f environment.yml```

# Acknowledgements
