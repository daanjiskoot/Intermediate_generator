import os
import sys
import time
import random
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, Descriptors, Draw, rdFMCS
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from selfies import encoder, decoder 


class _FingerprintCalculator:
    """
    Class to calculate the fingerprint for a molecule, given the fingerprint type.
    """
    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = 'get_' + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{fp_type} is not a supported fingerprint type.')
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return AllChem.Generate.Gen2DFingerprint(mol, AllChem.Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return AllChem.GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return AllChem.GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    """
    Fingerprint getter method. Fingerprint is returned after using object of 
    class '_FingerprintCalculator'
    """
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)


def get_fp_scores(smiles_back, target_smi, fp_type): 
    """
    Calculate the Tanimoto fingerprint (using fp_type fingerint) similarity between a list 
    of SMILES and a known target structure (target_smi). 
    """
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores
