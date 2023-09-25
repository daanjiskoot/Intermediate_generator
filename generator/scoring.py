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

from . import filters

from openeye import oechem, oeomega, oeshape 
from openeye.oeomega import OEOmega
from openeye.oechem import OEMol, OEParseSmiles, OEMolToSmiles


def tanimoto_complex(starting_smile, all_smiles, target_smile, exponent_path):
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
            similarity = 2 * (similarity1**exponent_path * similarity2**exponent_path) / (similarity1**exponent_path + similarity2**exponent_path)
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
    
    
def score_median_mols(starting_smile, target_smile, intermediates, exponent_path): 
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
    
    better_score = tanimoto_complex(starting_smile, all_smiles, target_smile, exponent_path)
    better_score = np.array(better_score)
    
    best_idx = better_score.argsort()[::-1]
    best_smi = [all_smiles[i] for i in best_idx]
    best_scores = [better_score[i] for i in best_idx]

    return best_smi, best_scores

import numpy as np
from openeye import oechem, oeshape, oeomega

def generate_best_conformer(smiles):
    # Create molecule from SMILES string
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        print("Couldn't parse smiles: %s" % smiles)
        return None

    # Generate conformers
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(200)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(False)

    if not omega(mol):
        print(f"Omega failed for SMILES: {smiles}")
        return None

    # Select the best conformer (i.e., the first one, as they're sorted by energy)
    mol = oechem.OEMol(mol.GetConf(oechem.OEHasConfIdx(0)))

    return mol

def rocs_old(starting_smile, all_smiles, target_smile, exponent_path):
    # Preparation
    prep = oeshape.OEOverlapPrep()
    start_mol = generate_best_conformer(starting_smile)
    prep.Prep(start_mol)  # prepares molecule for overlap (optimize, add hydrogens, etc.)

    target_mol = generate_best_conformer(target_smile)
    prep.Prep(target_mol)

    # Prepare list to store the valid SMILES strings and scores
    valid_smiles = []
    similarity_scores = []

    for smile in all_smiles:
        mol = generate_best_conformer(smile)
        if mol is not None:
            prep.Prep(mol)

            res1 = oeshape.OEROCSResult()
            oeshape.OEROCSOverlay(res1, start_mol, mol)
            score1 = res1.GetTanimotoCombo()  # get Tanimoto overlap score with start_mol
            score1 = score1 / 2
#            print("similarity path based to A", score1)

            res2 = oeshape.OEROCSResult()
            oeshape.OEROCSOverlay(res2, target_mol, mol)
            score2 = res2.GetTanimotoCombo()  # get Tanimoto overlap score with target_mol
            score2 = score2 / 2
#            print("similarity path based to B", score2)

            if score1 == 0 or score2 == 0:
                balanced_score = 0
            else:
                balanced_score = 2 * (score1**exponent_path * score2**exponent_path) / (score1**exponent_path + score2**exponent_path)
#                print("balanced score path based", balanced_score)

            valid_smiles.append(smile)
            similarity_scores.append(balanced_score)

    similarity_scores = np.array(similarity_scores)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Sort the smiles strings and scores using the sorted indices
    best_smiles = [valid_smiles[i] for i in sorted_indices]
    best_scores = [similarity_scores[i] for i in sorted_indices]

    return best_smiles, best_scores

def rocs2(starting_smile, intermediate_smile, target_smile, exponent_local_chemical_space):
    # Preparation
    prep = oeshape.OEOverlapPrep()

    start_mol = generate_best_conformer(starting_smile)
    if start_mol is None or start_mol.NumAtoms() == 0:
        print(f"Problem with starting SMILES: {starting_smile}")
        return
    prep.Prep(start_mol)  # prepares molecule for overlap (optimize, add hydrogens, etc.)

    target_mol = generate_best_conformer(target_smile)
    if target_mol is None or target_mol.NumAtoms() == 0:
        print(f"Problem with target SMILES: {target_smile}")
        return
    prep.Prep(target_mol)

    mol = generate_best_conformer(intermediate_smile)
    if mol is None or mol.NumAtoms() == 0:
        print(f"Problem with intermediate SMILES: {intermediate_smile}")
        return
    prep.Prep(mol)

    res1 = oeshape.OEROCSResult()
    oeshape.OEROCSOverlay(res1, start_mol, mol)
    score1 = res1.GetTanimotoCombo()  # get Tanimoto overlap score with start_mol
#    score1 = score1 / 2
    print("similarity local chemical space based to A", score1)

    res2 = oeshape.OEROCSResult()
    oeshape.OEROCSOverlay(res2, target_mol, mol)
    score2 = res2.GetTanimotoCombo()  # get Tanimoto overlap score with target_mol
#    score2 = score2 / 2
    print("similarity local chemical space based to B", score1)

    if score1 == 0 and score2 == 0:
        balanced_score = 0
    else:
        score1 = score1
        score2 = score2
        balanced_score = 2 * ((score1**exponent_local_chemical_space) * (score2**exponent_local_chemical_space)) / ((score1**exponent_local_chemical_space) + (score2**exponent_local_chemical_space))

    return balanced_score




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


def normalize_scores(scores):
    """Normalize a list of scores such that the maximum score maps to 0.5."""
    
    max_score = max(scores)
    
    # If the max score is 0, return the original scores (or handle accordingly)
    if max_score == 0:
        return [0] * len(scores)  
    
    normalized_scores = [score / (2 * max_score) for score in scores]
    return normalized_scores





# this function is in its test phase, please try with care
def score_lomap_tanimoto(liga_smiles, ligb_smiles, canon_smi_ls, exponent_local_chemical_space, contribution_lomap, contribution_similarity):
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
    lomap_scores = []
    tanimoto_scores = []
    liga_mol = Chem.MolFromSmiles(liga_smiles)
    ligb_mol = Chem.MolFromSmiles(ligb_smiles)
    
    for smiles in canon_smi_ls:
        current_mol = Chem.MolFromSmiles(smiles)
        try:
            lomap_score_am, lomap_score_mb = quantify_change(liga_mol, current_mol, ligb_mol)
            try:
                lomap_multiplied_score = 2 * ((lomap_score_am**exponent_local_chemical_space) * (lomap_score_mb**exponent_local_chemical_space)) / ((lomap_score_am**exponent_local_chemical_space) + (lomap_score_mb**exponent_local_chemical_space))
                print("lomap", lomap_multiplied_score)
                lomap_scores.append(lomap_multiplied_score)
            except ZeroDivisionError:
                lomap_multiplied_score = 0
                
            try: 
                tanimoto_multiplied_score = tanimoto_scoring(liga_smiles, smiles, ligb_smiles, exponent_local_chemical_space)
                tanimoto_scores.append(tanimoto_multiplied_score)
                print("tanimoto", tanimoto_multiplied_score)
            except ValueError:
                tanimoto_multiplied_score = 0
                
        except ValueError:
            continue

    # Normalize lomap and tanimoto scores
    lomap_scores = normalize_scores(lomap_scores)
    print("lomap_scores_normalized", lomap_scores)
    tanimoto_scores = normalize_scores(tanimoto_scores)
    print("taninoto scores normalized", tanimoto_scores)
    
    # Calculate multiplied score with normalized values
    for i in range(len(lomap_scores)):
        multiplied_score = (contribution_lomap * lomap_scores[i]) + (contribution_similarity * tanimoto_scores[i])
        print("multiplied", multiplied_score)
        smiles_dict[canon_smi_ls[i]] = multiplied_score

    sorted_smiles_dict = dict(sorted(smiles_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_smiles_dict)

    return sorted_smiles_dict


# exponent value in score_lomap_tanimoto and tanimoto_scoring should be the same!
# original function, the issue was the unnormalized values in extreme cases for generation
def score_lomap_tanimoto_old(liga_smiles, ligb_smiles, canon_smi_ls):
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
    exponent2 = 4
    liga_mol = Chem.MolFromSmiles(liga_smiles)
    ligb_mol = Chem.MolFromSmiles(ligb_smiles)
    
    for smiles in canon_smi_ls:
        current_mol = Chem.MolFromSmiles(smiles)
        try:
            lomap_score_am, lomap_score_mb = quantify_change(liga_mol, current_mol, ligb_mol)
            try:
                lomap_multiplied_score = 2 * ((lomap_score_am**exponent2) * (lomap_score_mb**exponent2)) / ((lomap_score_am**exponent2) + (lomap_score_mb**exponent2))
                print("lomap", lomap_multiplied_score)
            except ZeroDivisionError:
                lomap_multiplied_score = 0
                
            try: 
                tanimoto_multiplied_score = tanimoto_scoring(liga_smiles, smiles, ligb_smiles)
                print("tanimoto", tanimoto_multiplied_score)
            except ValueError:
                tanimoto_multiplied_score = 0
                
        except ValueError:
            continue

        multiplied_score = (0.9 * lomap_multiplied_score) + (0.1 * tanimoto_multiplied_score)
        print("multiplied", multiplied_score)
        smiles_dict[smiles] = multiplied_score

    sorted_smiles_dict = dict(sorted(smiles_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_smiles_dict


# exponent value in score_lomap_tanimoto and tanimoto_scoring should be the same!
def tanimoto_scoring(starting_smile, intermediate_smile, target_smile, exponent_local_chemical_space):
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

    similarity_score = 2 * (similarity1**exponent_local_chemical_space * similarity2**exponent_local_chemical_space) / (similarity1**exponent_local_chemical_space + similarity2**exponent_local_chemical_space)

    return similarity_score



# this function scores molecules using lomap and tanimoto. It suppresses output from the logger, that would give a lot of lomap info output. It also suppresses
# all rdkit warnings. Consider re-activating these warnings when improving/debugging/troubleshooting this part of the code. 
# this approach is primitive, basically supressing everything, consider just calling the scoring function if you do not care cluttered output.
def score_molecules_lomap_tanimoto(liga_smiles, ligb_smiles, generated_mols, exponent_local_chemical_space, contribution_lomap, contribution_similarity):
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
        sorted_smiles_dict = score_lomap_tanimoto(liga_smiles, ligb_smiles, generated_mols, exponent_local_chemical_space, contribution_lomap, contribution_similarity)
    finally:
       # restore output to default
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    return sorted_smiles_dict


# exponent value in score_lomap_tanimoto and tanimoto_scoring should be the same!
def score_lomap_rocs_old(liga_smiles, ligb_smiles, canon_smi_ls, omega, exponent_local_chemical_space):
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
    liga_mol = Chem.MolFromSmiles(liga_smiles)
    ligb_mol = Chem.MolFromSmiles(ligb_smiles)
    
    for smiles in canon_smi_ls:
        current_mol = Chem.MolFromSmiles(smiles)
        try:
            lomap_score_am, lomap_score_mb = quantify_change(liga_mol, current_mol, ligb_mol)
            try:
                lomap_multiplied_score = 2 * ((lomap_score_am**exponent_local_chemical_space) * (lomap_score_mb**exponent_local_chemical_space)) / ((lomap_score_am**exponent_local_chemical_space) + (lomap_score_mb**exponent_local_chemical_space))
                print("lomap", lomap_multiplied_score)
            except ZeroDivisionError:
                lomap_multiplied_score = 0
                
            try: 
                rocs_score = rocs_local(liga_smiles, smiles, ligb_smiles, omega, exponent_local_chemical_space)
                if rocs_score is None:
                    rocs_score = 0
                print("rocs", rocs_score)
                print(smiles)
            except ValueError:
                rocs_score = 0
                
        except ValueError:
            continue

        multiplied_score = (0.8 * lomap_multiplied_score) + (0.2 * rocs_score)
        print("multiplied", multiplied_score)
        smiles_dict[smiles] = multiplied_score

    sorted_smiles_dict = dict(sorted(smiles_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_smiles_dict

def score_lomap_rocs(liga_smiles, ligb_smiles, canon_smi_ls, omega, exponent_local_chemical_space, contribution_lomap, contribution_similarity):
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
    lomap_scores = []
    rocs_scores = []
    
    liga_mol = Chem.MolFromSmiles(liga_smiles)
    ligb_mol = Chem.MolFromSmiles(ligb_smiles)
    
    for smiles in canon_smi_ls:
        current_mol = Chem.MolFromSmiles(smiles)
        try:
            lomap_score_am, lomap_score_mb = quantify_change(liga_mol, current_mol, ligb_mol)
            try:
                lomap_multiplied_score = 2 * ((lomap_score_am**exponent_local_chemical_space) * (lomap_score_mb**exponent_local_chemical_space)) / ((lomap_score_am**exponent_local_chemical_space) + (lomap_score_mb**exponent_local_chemical_space))
#                print("lomap", lomap_multiplied_score)
                lomap_scores.append(lomap_multiplied_score)
            except ZeroDivisionError:
                lomap_multiplied_score = 0
                
            try: 
                rocs_score = rocs_local(liga_smiles, smiles, ligb_smiles, omega, exponent_local_chemical_space)
                if rocs_score is None:
                    rocs_score = 0
#                print("rocs", rocs_score)
                rocs_scores.append(rocs_score)
                print(smiles)
            except ValueError:
                rocs_score = 0
                
        except ValueError:
            continue

    # Normalize lomap and rocs scores
    lomap_scores = normalize_scores(lomap_scores)
#    print("lomap_scores_normalized", lomap_scores)
    rocs_scores = normalize_scores(rocs_scores)
#    print("rocs scores normalized", rocs_scores)
    
    # Calculate multiplied score with normalized values
    for i in range(len(lomap_scores)):
        multiplied_score = (contribution_lomap * lomap_scores[i]) + (contribution_similarity * rocs_scores[i])
#        print("multiplied", multiplied_score)
        smiles_dict[canon_smi_ls[i]] = multiplied_score

    sorted_smiles_dict = dict(sorted(smiles_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_smiles_dict


# this function scores molecules using lomap and tanimoto. It suppresses output from the logger, that would give a lot of lomap info output. It also suppresses
# all rdkit warnings. Consider re-activating these warnings when improving/debugging/troubleshooting this part of the code. 
# this approach is primitive, basically supressing everything, consider just calling the scoring function if you do not care cluttered output.
def score_molecules_lomap_rocs(liga_smiles, ligb_smiles, generated_mols, exponent_local_chemical_space, contribution_lomap, contribution_similarity):
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
    omega = initialize_omega()
    
    # redirect output to null
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        # call function
        sorted_smiles_dict = score_lomap_rocs(liga_smiles, ligb_smiles, generated_mols, omega, exponent_local_chemical_space, contribution_lomap, contribution_similarity)
    finally:
        # restore output to default
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    print("you are succesfully using ROCS scoring")
    return sorted_smiles_dict

def initialize_omega():
    omega = OEOmega()
    omega.SetMaxConfs(100)
    omega.SetStrictStereo(False)
    omega.SetStrictAtomTypes(False)
    return omega

def AnalogMolInitial(smi):
    mol = OEMol()
    OEParseSmiles(mol, smi)
    return mol

def generate_best_conformer(smiles_string):
    omega = initialize_omega()  # Use the common initialization function
    
    mol = oechem.OEMol()
    if not oechem.OEParseSmiles(mol, smiles_string):
        return None
    if not omega(mol):
        return None

    return mol

def process_analog_smi_list_to_smiles(analog_smi_list, noconfs, refmol, omega):
    tancombolist = []
    shapetan = []
    colortan = []
    smiles_list = []

    for i in range(len(analog_smi_list)):
        fitmol = AnalogMolInitial(analog_smi_list[i])
        options = oeshape.OEROCSOptions()
        options.SetNumBestHits(noconfs)
        options.SetConfsPerHit(noconfs)
        rocs = oeshape.OEROCS(options)
        omega(fitmol)
        fitmol.SetTitle(f'AnalogNum{i}')
        rocs.AddMolecule(fitmol)

        for res in rocs.Overlay(refmol):
            outmol = res.GetOverlayConfs()
            oeshape.OERemoveColorAtoms(outmol)
            oechem.OEAddExplicitHydrogens(outmol)
            smiles_representation = OEMolToSmiles(outmol)
            smiles_list.append(smiles_representation)
            tancombolist.append(res.GetTanimotoCombo())
            shapetan.append(res.GetShapeTanimoto())
            colortan.append(res.GetColorTanimoto())

    return tancombolist, shapetan, colortan, smiles_list


def rocs(starting_smile, all_smiles, target_smile, noconfs, omega, exponent_path):
    start_mol = generate_best_conformer(starting_smile)
    target_mol = generate_best_conformer(target_smile)
    
    # testing to fix the 1 mol with 10 n-rounds:
    all_smiles = filters.drop_duplicates(starting_smile, target_smile, all_smiles)
    # Process the analog SMILES list using starting_smile
    tancombolist_start, _, _, _ = process_analog_smi_list_to_smiles(all_smiles, noconfs, start_mol, omega)
    tancombolist_start = [x / 2 for x in tancombolist_start]
    # we devide by 2 because this consists of both shape and color, which range individually between 0 and 1, allowing for their 
    # combo value to get up to 2. Therefore /2 normalizes between [0,1].
#    print(tancombolist_start, "tancombolist start")
    
    # Process the analog SMILES list using target_smile
    tancombolist_target, _, _, processed_smiles_list = process_analog_smi_list_to_smiles(all_smiles, noconfs, target_mol, omega)
    tancombolist_target = [x / 2 for x in tancombolist_target]
    # we devide by 2 because this consists of both shape and color, which range individually between 0 and 1, allowing for their 
    # combo value to get up to 2. Therefore /2 normalizes between [0,1].
#    print(tancombolist_target, "tancombolist target")

    # Directly computing the balanced scores within the list comprehension
    balanced_scores = []
    for score1, score2 in zip(tancombolist_start, tancombolist_target):
        if score1 == 0 or score2 == 0:
            balanced_score = 0
        else:
            balanced_score = (2 * (score1**exponent_path * score2**exponent_path)) / (score1**exponent_path + score2**exponent_path)
#           print("balanced score path based rocs", balanced_score)
        balanced_scores.append(balanced_score)
    
    # Sorting the scores
    sorted_indices = np.argsort(balanced_scores)[::-1]
    best_smiles = [processed_smiles_list[i] for i in sorted_indices]
    best_scores = [balanced_scores[i] for i in sorted_indices]

    return best_smiles, best_scores

def rocs_local(starting_smile, intermediate_smile, target_smile, omega, exponent_local_chemical_space):
    # Preparation
    prep = oeshape.OEOverlapPrep()

    start_mol = generate_best_conformer(starting_smile)
    if start_mol is None or start_mol.NumAtoms() == 0:
        print(f"Problem with starting SMILES: {starting_smile}")
        return None
    prep.Prep(start_mol)  # prepares molecule for overlap (optimize, add hydrogens, etc.)

    target_mol = generate_best_conformer(target_smile)
    if target_mol is None or target_mol.NumAtoms() == 0:
        print(f"Problem with target SMILES: {target_smile}")
        return None
    prep.Prep(target_mol)

    # Process the intermediate SMILES
    tancombolist_start, _, _, _ = process_analog_smi_list_to_smiles([intermediate_smile], 1, start_mol, omega)
    tancombolist_target, _, _, _ = process_analog_smi_list_to_smiles([intermediate_smile], 1, target_mol, omega)
    
    if not tancombolist_start or not tancombolist_target:
        print("Error processing intermediate SMILES for ROCS.")
        return None

    score1 = tancombolist_start[0] / 2  # normalization
    print("similarity local chemical space based to A", score1)
    
    score2 = tancombolist_target[0] / 2  # normalization
    print("similarity local chemical space based to B", score2)

    # Compute balanced score
    if score1 == 0 and score2 == 0:
        balanced_score = 0
    else:
        balanced_score = 2 * ((score1**exponent_local_chemical_space) * (score2**exponent_local_chemical_space)) / ((score1**exponent_local_chemical_space) + (score2**exponent_local_chemical_space))
 #       print("balanced score local chemical space", balanced_score)

    return balanced_score
