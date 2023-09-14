#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import random
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import contextlib
import logging
import openbabel
from openbabel import pybel
import rdkit

from random import randrange
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import AllChem, rdMolAlign, Descriptors, Draw
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdFMCS, MolFromSmarts
from selfies import encoder, decoder 
from rdkit.Chem.Draw import MolToImage, IPythonConsole, rdMolDraw2D

import lomap

from . import fingerprint
from . import generation
from . import filters
from . import scoring
from . import visualization

# Ignore warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# note to user: warning messages of the scoring function of the local chemical space are suppressed.

def intermediates(
    perts_to_intrap, 
    lig_path,
    num_tries, 
    num_random_smiles,
    collect_bidirectional,
    fp_type,
    num_random_samples,
    num_mutation_ls,
    n_rounds,
    exponent_path,
    exponent_local_chemical_space,
    sdf,
    svg,
    png,
    scoring_method,
    contribution_lomap,
    contribution_similarity,
    base_dir
):
    for liga, ligb in perts_to_intrap:
        start_time = time.time()

        # Check if the input is molecule name, actual molecule object, or file path
        if isinstance(liga, str):
            if liga.endswith('.sdf'):
                liga = read_molecule_from_sdf(liga)
            elif liga.endswith('.mol2'):
                liga = visualization.mol2_to_rdkit(liga)
            else:
                liga = visualization.fetch_molecule(liga, lig_path, ".sdf") or visualization.fetch_molecule(liga, lig_path, ".mol2")

        if isinstance(ligb, str):
            if ligb.endswith('.sdf'):
                ligb = read_molecule_from_sdf(ligb)
            elif ligb.endswith('.mol2'):
                ligb = visualization.mol2_to_rdkit(ligb)
            else:
                ligb = visualization.fetch_molecule(ligb, lig_path, ".sdf") or visualization.fetch_molecule(ligb, lig_path, ".mol2")

        if liga is None or ligb is None:
            raise ValueError("Failed to fetch or read molecules.")

        liga_smiles = mol2smi(liga)
        ligb_smiles = mol2smi(ligb)
        
        if scoring_method == "3D":
            generated_paths = generation.generate_multiple_paths_rocs(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, n_rounds=n_rounds, fp_type=fp_type)
        else: 
            generated_paths = generation.generate_multiple_paths(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, n_rounds=n_rounds, fp_type=fp_type)
            
        generated_mols = generation.generate_chemical_space(liga_smiles, ligb_smiles, generated_paths, num_random_samples, num_mutation_ls, fp_type=fp_type)
        
        if scoring_method == "3D":
            sorted_smiles_dict = scoring.score_molecules_lomap_rocs(liga_smiles, ligb_smiles, generated_mols, exponent_local_chemical_space, contribution_lomap, contribution_similarity)
        else:
            sorted_smiles_dict = scoring.score_molecules_lomap_tanimoto(liga_smiles, ligb_smiles, generated_mols, exponent_local_chemical_space, contribution_lomap, contribution_similarity)
            
        selected_intermediate = next(iter(sorted_smiles_dict))
        filename = f'{tgt}_{pert}'
        filepath = f'{base_dir}/{filename}'
        filepath_intermediate = f'{base_dir}/intermediate_{filename}.sdf'
        
        if sdf == True:
            visualization.align_intermediates_to_references(liga, ligb, selected_intermediate, filepath_intermediate, n_conformers=10)
        
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        
        visualization.visualise(liga_smiles, ligb_smiles, selected_intermediate, mins, secs, filepath, filename, svg=svg, png=png)