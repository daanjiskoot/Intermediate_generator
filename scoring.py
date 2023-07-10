import rdkit
import lomap
import os
import sys
import selfies
import random
import numpy as np

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi, AllChem, rdMolAlign, Descriptors, Draw
from selfies import encoder, decoder 


def tanimoto_complex(starting_smile, all_smiles, target_smile, exponent=4):
    """
    Calculate the Tanimoto similarity for a list of molecules to a start and target molecule.

    Args:
        starting_smile (str): SMILES string of the starting molecule.
        all_smiles (list): List of SMILES strings of the molecules.
        target_smile (str): SMILES string of the target molecule.

    Returns:
        list: List of similarity scores.
    """
    
    # put all stuff for the start en target outside of loop, all_smiles in loop because multiple molecules, using ECFP4
    similarity_score = []
    start = Chem.MolFromSmiles(starting_smile)
    target = Chem.MolFromSmiles(target_smile)
    # calculate the ECFP4 fingerprints for the three molecules
    fp_1 = AllChem.GetMorganFingerprint(start, radius=2)
    #fp_2 = AllChem.GetMorganFingerprint(intermediate, radius=3)
    fp_3 = AllChem.GetMorganFingerprint(target, radius=2)

    for item in all_smiles: 
        mol    = Chem.MolFromSmiles(item)
        if mol is None:
            continue
        fp_mol = AllChem.GetMorganFingerprint(mol, radius=2)
        similarity1  = DataStructs.TanimotoSimilarity(fp_mol, fp_1)
        similarity2  = DataStructs.TanimotoSimilarity(fp_mol, fp_3)
        # Calculate the harmonic mean of the similarities
        if similarity1 == 0 or similarity2 == 0:
            similarity = 0
        else:
            similarity = 2 * (similarity1**exponent * similarity2**exponent) / (similarity1**exponent + similarity2**exponent)
        similarity_score.append(similarity)

    return similarity_score

def sanitize_smiles(smi):
    """
    Sanitize and canonicalize a SMILES string.

    Args:
        smi (str): SMILES string to sanitize.

    Returns:
        tuple: A molecule object, a canonical SMILES string, and a boolean indicating success.
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=True, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
    
def score_median_mols(starting_smile, target_smile, intermediates): 
    """
    Score the median molecules.

    Args:
        starting_smile (str): SMILES string of the starting molecule.
        target_smile (str): SMILES string of the target molecule.
        intermediates (list): List of SMILES strings of the intermediate molecules.

    Returns:
        tuple: List of best SMILES strings and their scores.
    """
    
    all_smiles = [] # Collection of valid smile strings 
    for smi in intermediates: 
        if Chem.MolFromSmiles(smi) != None: 
            mol, smi_canon, _ = sanitize_smiles(smi)
            all_smiles.append(smi_canon)

    all_smiles = list(set(all_smiles))
    
    better_score = tanimoto_complex(starting_smile, all_smiles, target_smile)
    better_score = np.array(better_score)
    
    best_idx = better_score.argsort()[::-1]
    best_smi = [all_smiles[i] for i in best_idx]
    best_scores = [better_score[i] for i in best_idx]

    return best_smi, best_scores


def computeLOMAPScore(lig1, lig2):
    """Computes the LOMAP score for two input ligands, see https://github.com/OpenFreeEnergy/Lomap/blob/main/lomap/mcs.py."""
    """
    Computes the LOMAP score for two input molecules.

    Args:
        lig1 (rdkit.Chem.rdchem.Mol): First molecule.
        lig2 (rdkit.Chem.rdchem.Mol): Second molecule.

    Returns:
        float: The computed LOMAP score.
    """

    AllChem.EmbedMolecule(lig1, useRandomCoords=True)
    AllChem.EmbedMolecule(lig2, useRandomCoords=True)

    MC = lomap.MCS(lig1, lig2, time=2, verbose=None)

    # # Rules calculations
    mcsr = MC.mcsr()
    strict = MC.tmcsr(strict_flag=True)
    loose = MC.tmcsr(strict_flag=False)
    mncar = MC.mncar()
    atnum = MC.atomic_number_rule()
    hybrid = MC.hybridization_rule()
    sulf = MC.sulfonamides_rule()
    het = MC.heterocycles_rule()
    growring = MC.transmuting_methyl_into_ring_rule()
    changering = MC.transmuting_ring_sizes_rule()


    score = mncar * mcsr * atnum * hybrid
    score *= sulf * het * growring
    lomap_score = score*changering

    return lomap_score

def quantify_change(liga, median, ligb):
    """
    Quantify the change from a starting molecule to a target molecule via a median molecule.

    Args:
        liga (rdkit.Chem.rdchem.Mol): The starting molecule.
        median (rdkit.Chem.rdchem.Mol): The median molecule.
        ligb (rdkit.Chem.rdchem.Mol): The target molecule.

    Returns:
        tuple: A pair of LOMAP scores.
    """


    lomap_score_am = computeLOMAPScore(liga, median)
    lomap_score_mb = computeLOMAPScore(median, ligb)
    
    
    return lomap_score_am,lomap_score_mb




# exponent value in score_lomap_tanimoto and tanimoto_scoring should be the same!
def score_lomap_tanimoto(liga_smiles, ligb_smiles, canon_smi_ls):
    """
    Calculate LOMAP scores for a list of molecules.

    Args:
        liga_smiles (str): SMILES string of the starting molecule.
        ligb_smiles (str): SMILES string of the target molecule.
        canon_smi_ls (list): List of canonical SMILES strings of the intermediate molecules.

    Returns:
        dict: Dictionary of canonical SMILES strings and their corresponding LOMAP scores.
    """
    
    smiles_dict = {}
    exponent2 = 2
    liga_mol = Chem.MolFromSmiles(liga_smiles)
    ligb_mol = Chem.MolFromSmiles(ligb_smiles)
    
    for smiles in canon_smi_ls:
        current_mol = Chem.MolFromSmiles(smiles)
        try:
            lomap_score_am, lomap_score_mb = quantify_change(liga_mol, current_mol, ligb_mol)
            try:
                lomap_multiplied_score = 2 * ((lomap_score_am**exponent2) * (lomap_score_mb**exponent2)) / ((lomap_score_am**exponent2) + (lomap_score_mb**exponent2))
                #print("lomap", lomap_multiplied_score)
            except ZeroDivisionError:
                lomap_multiplied_score = 0
                
            try: 
                tanimoto_multiplied_score = tanimoto_scoring(liga_smiles, smiles, ligb_smiles)
                #print("tanimoto", tanimoto_multiplied_score)
            except ValueError:
                tanimoto_multiplied_score = 0
                
        except ValueError:
            continue

        multiplied_score = (0.9 * lomap_multiplied_score) + (0.1 * tanimoto_multiplied_score)
        #print("multiplied", multiplied_score)
        smiles_dict[smiles] = multiplied_score

    sorted_smiles_dict = dict(sorted(smiles_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_smiles_dict


# exponent value in score_lomap_tanimoto and tanimoto_scoring should be the same!
def tanimoto_scoring(starting_smile, intermediate_smile, target_smile, exponent=2):
    """
    Calculate the Tanimoto similarity for a single molecule to a start and target molecule.

    Args:
        starting_smile (str): SMILES string of the starting molecule.
        intermediate_smile (str): SMILES string of the intermediate molecule.
        target_smile (str): SMILES string of the target molecule.
        exponent (int): Exponent for the weighted sum calculation.

    Returns:
        float: Harmonic mean of the Tanimoto similarity scores.
    """
    # Put all stuff for the start and target outside of loop,
    # Using ECFP4 fingerprints
    start = Chem.MolFromSmiles(starting_smile)
    target = Chem.MolFromSmiles(target_smile)
    intermediate = Chem.MolFromSmiles(intermediate_smile)

    # Calculate the ECFP4 fingerprints for the three molecules
    fp_1 = AllChem.GetMorganFingerprint(start, radius=2)
    fp_3 = AllChem.GetMorganFingerprint(target, radius=2)
    fp_mol = AllChem.GetMorganFingerprint(intermediate, radius=2)

    # Calculate the Tanimoto similarity
    similarity1  = DataStructs.TanimotoSimilarity(fp_mol, fp_1)
    similarity2  = DataStructs.TanimotoSimilarity(fp_mol, fp_3)
    
    # Calculate the harmonic mean of the similarities

    similarity_score = 2 * (similarity1**exponent * similarity2**exponent) / (similarity1**exponent + similarity2**exponent)

    return similarity_score



# this function scores molecules using lomap and tanimoto. It suppresses output from the logger, that would give a lot of lomap info output. It also suppresses
# all rdkit warnings. Consider re-activating these warnings when improving/debugging/troubleshooting this part of the code. 
# this approach is primitive, basically supressing everything, consider just calling the scoring function if you do not care cluttered output.
def score_molecules_lomap_tanimoto(liga_smiles, ligb_smiles, generated_mols):
    """
    Scores molecules using LOMAP and Tanimoto, while suppressing output from the logger, including any LOMAP info or RDKit warnings.
    This function may be modified to reactivate these warnings for debugging or troubleshooting. 
    this approach is quite thorough, basically supressing everything, consider calling just the scoring function if you do not care about cluttered output.
    
    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        generated_mols (list): List of generated molecules to be scored.

    Returns:
        dict: The dictionary with scored molecules sorted by scores.
    """
    
    # redirect output to null
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        # call your function
        sorted_smiles_dict = score_lomap_tanimoto(liga_smiles, ligb_smiles, generated_mols)
    finally:
        # restore output to default
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    return sorted_smiles_dict