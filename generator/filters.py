import numpy as np
import random
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit import DataStructs
from rdkit.Chem.rdchem import ChiralType

def flatten(nested_list):
    """
    Recursively flattens a nested list.
    
    Args:
    nested_list (list): A list that may contain other lists as elements.
    
    Returns:
    list: A flattened list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def count_heavy_atoms(mol):
    """
    Count the number of heavy atoms in a molecule.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The molecule.
    
    Returns:
    int: The number of heavy atoms in the molecule.
    """
    return sum([1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])


def filter_smiles_by_heavy_atoms(reference_smiles_1, reference_smiles_2, smiles_list):
    """
    Filter a list of SMILES strings based on the number of heavy atoms in two reference molecules.
    
    Args:
    reference_smiles_1, reference_smiles_2 (str): SMILES strings of the reference molecules.
    smiles_list (list of str): The list of SMILES strings to filter.
    
    Returns:
    list of str: The filtered list of SMILES strings.
    """
    # Calculate heavy atom counts for liga and ligb
    liga_mol = Chem.MolFromSmiles(reference_smiles_1)
    ligb_mol = Chem.MolFromSmiles(reference_smiles_2)
    liga_heavy_atoms = count_heavy_atoms(liga_mol)
    ligb_heavy_atoms = count_heavy_atoms(ligb_mol)

    # Select the lower and higher of the two heavy atom counts
    min_heavy_atoms = min(liga_heavy_atoms, ligb_heavy_atoms)
    max_heavy_atoms = max(liga_heavy_atoms, ligb_heavy_atoms)

    # Calculate the 70% and 110% thresholds
    lower_threshold = 0.7 * min_heavy_atoms
    upper_threshold = 1.1 * max_heavy_atoms

    # Filter canon_smi_ls based on the thresholds
    filtered_canon_smi_ls = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        heavy_atoms = count_heavy_atoms(mol)
        if heavy_atoms >= lower_threshold and heavy_atoms <= upper_threshold:
            filtered_canon_smi_ls.append(smiles)

    return filtered_canon_smi_ls

def find_mcs(smiles_1, smiles_2):
    """
    Find the maximum common substructure (MCS) of two molecules.
    
    Args:
    smiles_1, smiles_2 (str): SMILES strings of the two molecules.
    
    Returns:
    rdkit.Chem.rdchem.Mol: The MCS of the two molecules.
    """
    mol_a = Chem.MolFromSmiles(smiles_1)
    mol_b = Chem.MolFromSmiles(smiles_2)

    mcs = rdFMCS.FindMCS([mol_a, mol_b],
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny,
                         matchValences=False,
                         ringMatchesRingOnly=True,
                         completeRingsOnly=True,
                         matchChiralTag=False,
                         timeout=2)
    template = Chem.MolFromSmarts(mcs.smartsString)

    return template

def find_large_mcs(mol_a, mol_b):
    """
    Finds the maximum common substructure (MCS) between two molecules.

    Parameters
    ----------
    mol_a, mol_b : rdkit.Chem.rdchem.Mol
        Molecules for comparison.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A molecule representing the maximum common substructure between mol_a and mol_b.
    """


    mcs = rdFMCS.FindMCS([mol_a, mol_b],
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny,
                         matchValences=False,
                         ringMatchesRingOnly=False,
                         completeRingsOnly=False,
                         matchChiralTag=False,
                         timeout=2)
    template = Chem.MolFromSmarts(mcs.smartsString)

    return template


def find_mcs_3_mols(smiles_a, smiles_b, intermediate_smiles):
    """
    Finds the maximum common substructure (MCS) among three molecules.

    Parameters
    ----------
    smiles_a : str
        SMILES representation of the first molecule for comparison.
    smiles_b : str
        SMILES representation of the second molecule for comparison.
    intermediate_smiles : str
        SMILES representation of the intermediate molecule for comparison.

    Returns
    -------
    template : rdkit.Chem.rdchem.Mol
        Molecule representing the maximum common substructure.
    """
    
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    mol_intermediate = Chem.MolFromSmiles(intermediate_smiles)

    mcs = rdFMCS.FindMCS([mol_a, mol_b, mol_intermediate],
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny,
                         matchValences=False,
                         ringMatchesRingOnly=True,
                         completeRingsOnly=True,
                         matchChiralTag=False,
                         timeout=2)
    template = Chem.MolFromSmarts(mcs.smartsString)
    
    return template


def get_positive_charge(mol):
    """
    Calculates the total positive charge of a molecule.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The molecule to calculate charge for.
    
    Returns:
    int: The total positive charge of the molecule.
    """
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)

def get_negative_charge(mol):
    """
    Calculates the total negative charge of a molecule.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The molecule to calculate charge for.
    
    Returns:
    int: The total negative charge of the molecule.
    """
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)

def filter_charge(liga, ligb, mol_list):
    """
    Filters a list of molecules based on their charge. The molecules with charges within the range of the charges of the 
    two reference molecules are kept.
    
    Args:
    liga, ligb (rdkit.Chem.rdchem.Mol): The two reference molecules.
    mol_list (list of rdkit.Chem.rdchem.Mol): The list of molecules to filter.
    
    Returns:
    list of rdkit.Chem.rdchem.Mol: The filtered list of molecules.
    """
    # Calculate charges of starting molecules
    pos_charge_liga = get_positive_charge(liga)
    pos_charge_ligb = get_positive_charge(ligb)
    neg_charge_liga = get_negative_charge(liga)
    neg_charge_ligb = get_negative_charge(ligb)

    # Calculate min and max charge
    min_pos_charge = min(pos_charge_liga, pos_charge_ligb)
    max_pos_charge = max(pos_charge_liga, pos_charge_ligb)
    min_neg_charge = min(neg_charge_liga, neg_charge_ligb)
    max_neg_charge = max(neg_charge_liga, neg_charge_ligb)

    # Filter molecules
    filtered_mols = [mol for mol in mol_list if min_pos_charge <= get_positive_charge(mol) <= max_pos_charge and min_neg_charge <= get_negative_charge(mol) <= max_neg_charge]

    return filtered_mols

# check for removal
#def drop_duplicates(liga_smiles, ligb_smiles, molecule_list):
#    """
#    Processes a list of molecules, removing any duplicates and the two specified input molecules.
#
#    Args:
#        liga_smiles (str): SMILES string of the first input molecule.
#        ligb_smiles (str): SMILES string of the second input molecule.
#        molecule_list (list): List of SMILES strings of the molecules.
#
#    Returns:
#
#    list: List of unique SMILES strings, excluding the input molecules.
#    """
#    molecule_list = list(set(molecule_list))
#
#    # Remove the input molecules from the list if they are present
#    # and list has more than one item.
#    if len(molecule_list) > 1 and liga_smiles in molecule_list:
#        molecule_list.remove(liga_smiles)
#
#    if len(molecule_list) > 1 and ligb_smiles in molecule_list:
#        molecule_list.remove(ligb_smiles)
#
 #   return molecule_list

# check if this function functions appropriately
def drop_duplicates(liga_smiles, ligb_smiles, molecule_list):
    molecule_list = flatten(molecule_list)
    new_molecule_list = []
    duplicates = set()
    for molecule in molecule_list:
        if molecule not in duplicates:
            new_molecule_list.append(molecule)
            duplicates.add(molecule)
    # Remove the input molecules from the list if they are present
    # and list has more than one item.
    if len(new_molecule_list) > 1 and liga_smiles in new_molecule_list:
        new_molecule_list.remove(liga_smiles)
    if len(new_molecule_list) > 1 and ligb_smiles in new_molecule_list:
        new_molecule_list.remove(ligb_smiles)
    return new_molecule_list



def get_heavy_atoms(mol):
    """
    Get all different types of heavy atoms in a molecule
    """
    return set(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1) 

def drop_new_atom_type(liga, ligb, mol_list):
    """
    Filter the molecules in mol_list which contain an atom not present in liga and ligb
    """
    # Collect all heavy atoms from liga and ligb
    heavy_atoms = get_heavy_atoms(liga) | get_heavy_atoms(ligb)

    # Filter the molecules
    filtered_mol_list = []
    for mol in mol_list:
        mol_atoms = get_heavy_atoms(mol)
        if mol_atoms.issubset(heavy_atoms):
            filtered_mol_list.append(mol)
    
    return filtered_mol_list


# differently sized ring filter
def get_ring_sizes(mol):
    """
    Extracts ring sizes from the input molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Input molecule

    Returns:
        list: Sorted list of ring sizes in the molecule
    """
    ring_info = mol.GetRingInfo()
    return sorted(len(ring) for ring in ring_info.AtomRings())

def remove_divergent_ring_sizes(liga, ligb, mols):
    """
    Filters out molecules from the list that have ring sizes not found in either reference molecule.

    Args:
        liga (rdkit.Chem.rdchem.Mol): First reference molecule
        ligb (rdkit.Chem.rdchem.Mol): Second reference molecule
        mols (list): List of molecules to filter

    Returns:
        list: List of filtered molecules
    """
    # Get the ring sizes in the reference molecules
    ref_ring_sizes = set(get_ring_sizes(liga))
    ref_ring_sizes.update(get_ring_sizes(ligb))

    # Filter the list of molecules
    return [mol for mol in mols if set(get_ring_sizes(mol)).issubset(ref_ring_sizes)]



def has_lone_pair(mol):
    """
    Checks if the input molecule contains any atoms with lone pairs.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Input molecule

    Returns:
        bool: True if the molecule contains an atom with a lone pair, False otherwise
    """    
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False

def lone_pair(liga, ligb, mols):
    """
    Filters out molecules from the list that have lone pairs if neither of the reference molecules have one.

    Args:
        liga (rdkit.Chem.rdchem.Mol): First reference molecule
        ligb (rdkit.Chem.rdchem.Mol): Second reference molecule
        mols (list): List of molecules to filter

    Returns:
        list: List of filtered molecules
    """
    # Check if reference molecules have lone pairs
    if not (has_lone_pair(liga) or has_lone_pair(ligb)):
        # If neither has a lone pair, remove molecules with lone pairs
        return [mol for mol in mols if not has_lone_pair(mol)]
    else:
        # If at least one has a lone pair, don't remove any molecules
        return mols

    

# this function applies a series of filters on the molecules that are generated in the path-based generation.
# !!perhaps change the order of operations to increase efficiency of the code.

def filters_path_based_generation(liga_smiles, ligb_smiles, mols):
    """
    Applies a series of filters on the molecules that are generated in the path-based generation.

    Args:
        liga_smiles (str): SMILES representation of the first reference molecule
        ligb_smiles (str): SMILES representation of the second reference molecule
        mols (list): List of molecules to filter

    Returns:
        list: List of filtered molecules in SMILES format
    """
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    
    # check if all generated molecules contain the MCS. 
    mcs = find_mcs(liga_smiles, ligb_smiles)

    # Create a new list of molecules that contain the MCS
    mols_generated_paths = [Chem.MolFromSmiles(smi) for smi in mols]
    mols_with_mcs = [mol for mol in mols_generated_paths if mol.HasSubstructMatch(mcs)]

    # Revert to SMILES
    mols_with_mcs_smiles = [Chem.MolToSmiles(mol) for mol in mols_with_mcs]
    
    # check if the chirality matches
    filtered_list_mcs_chirality = chiral_equal(liga, ligb, mols_with_mcs)
    mols_with_mcs_smiles_chiral = [Chem.MolToSmiles(mol) for mol in filtered_list_mcs_chirality]

    # filter based on number of heavy atoms
    num_heavy_atom = filter_smiles_by_heavy_atoms(liga_smiles, ligb_smiles, mols_with_mcs_smiles_chiral)
    num_heavy_atom_mols = [Chem.MolFromSmiles(mol) for mol in num_heavy_atom]

    # filter molecules with a divergent charge count
    charge_check_mols = filter_charge(liga, ligb, num_heavy_atom_mols)
    charge_check_smiles = [Chem.MolToSmiles(smi) for smi in charge_check_mols]

    # heavy atom filter. This would register all heavy atoms in the original molecules, and if there is a novel heavy atom introduced, we throw it out. 
    # unnessacery in this step, because we are in the limited chemical space between 2 molecules
   # drop_introduced_atoms = drop_new_atom_type(liga, ligb, charge_check_mols)

    # divergent ring size removal filter
    same_rings = remove_divergent_ring_sizes(liga, ligb, charge_check_mols)

    # remove molecules with lone pair if not present in starting molecules
    lone_pair_removed = lone_pair(liga, ligb, same_rings)
    lone_pair_removed_smiles = [Chem.MolToSmiles(smi) for smi in lone_pair_removed]

    return lone_pair_removed_smiles


# this function applies a series of filters on the molecules that are generated in the local space generation.
# !!perhaps change the order of operations to increase efficiency of the code.

def filters_local_chemical_space_generation(liga_smiles, ligb_smiles, mols):
    """
    Applies a series of filters on the molecules that are generated in the local chemical space generation.

    Args:
        liga_smiles (str): SMILES representation of the first reference molecule
        ligb_smiles (str): SMILES representation of the second reference molecule
        mols (list): List of molecules to filter

    Returns:
        list: List of filtered molecules in SMILES format
    """
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    
    # filter based on number of heavy atoms
    heavy_atom_filter = filter_smiles_by_heavy_atoms(liga_smiles, ligb_smiles, mols)

    # apply overlapping chirality filter
    chiral_mols = [Chem.MolFromSmiles(mol) for mol in heavy_atom_filter]
    chirality_filter = chiral_equal(liga, ligb, chiral_mols)
    chiral_smiles = [Chem.MolToSmiles(mol) for mol in chirality_filter]

    # filter based on mcs
    mcs = find_mcs(liga_smiles, ligb_smiles)
    test_mols = [Chem.MolFromSmiles(smi) for smi in chiral_smiles]
    mcs_mols = [mol for mol in test_mols if mol.HasSubstructMatch(mcs)]
    mcs_smiles = [Chem.MolToSmiles(smi) for smi in mcs_mols]

    # filter molecules with a divergent charge count    
    charge_corrected_mols = filter_charge(liga, ligb, mcs_mols)
    charge_corrected_smiles = [Chem.MolToSmiles(smi) for smi in charge_corrected_mols]

    # think about the introduction of a novel heavy atom filter. This would register all heavy atoms in the original molecules, and if there is a novel heavy 
    # atom introduced, we throw it out. 
    dropped_new_atoms = drop_new_atom_type(liga, ligb, charge_corrected_mols)
    dropped_atom_type_smiles = [Chem.MolToSmiles(smi) for smi in dropped_new_atoms]

    # alternative sized ring filter 
    same_rings_remain = remove_divergent_ring_sizes(liga, ligb, dropped_new_atoms)
    same_rings_remain_smiles = [Chem.MolToSmiles(smi) for smi in same_rings_remain]

    # remove molecules with lone pair if not present in starting molecules
    remove_lone_pair = lone_pair(liga, ligb, same_rings_remain)
    remove_lone_pair_smiles = [Chem.MolToSmiles(smi) for smi in remove_lone_pair]

    # drop duplicates liga and ligb
    filtered_smiles = drop_duplicates(liga_smiles, ligb_smiles, remove_lone_pair_smiles)

    return filtered_smiles

# check if the new findpotentialstereo works more exhaustively
def count_chiral_atoms(mol):
    """
    Counts the number of chiral centers (CW and CCW) in the input molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Input molecule

    Returns:
        tuple: The number of CW and CCW chiral centers in the molecule
    """
    # this is the old way of assinging stereochemistry
   # Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
# this function should be more precise and quicker
    Chem.FindPotentialStereo(mol, cleanIt=True)
    
    chiral_centers_cw = 0
    chiral_centers_ccw = 0
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW:
            chiral_centers_cw += 1
        elif atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral_centers_ccw += 1
    return chiral_centers_cw, chiral_centers_ccw

def chiral_equal(mol1, mol2, mol_list):
    """
    Filters out molecules from the list that have different chirality counts from the reference molecules.

    Args:
        mol1 (rdkit.Chem.rdchem.Mol): First reference molecule
        mol2 (rdkit.Chem.rdchem.Mol): Second reference molecule
        mol_list (list): List of molecules to filter

    Returns:
        list: List of filtered molecules
    """
    chiral_count1_cw, chiral_count1_ccw = count_chiral_atoms(mol1)
    chiral_count2_cw, chiral_count2_ccw = count_chiral_atoms(mol2)

    min_chiral_cw = min(chiral_count1_cw, chiral_count2_cw)
    max_chiral_cw = max(chiral_count1_cw, chiral_count2_cw)
    min_chiral_ccw = min(chiral_count1_ccw, chiral_count2_ccw)
    max_chiral_ccw = max(chiral_count1_ccw, chiral_count2_ccw)

    filtered_list = []
    for mol in mol_list:
        chiral_count_cw, chiral_count_ccw = count_chiral_atoms(mol)
        
        if chiral_count1_cw == chiral_count2_cw == 0:
            if chiral_count_cw != 0:
                continue
        elif not min_chiral_cw <= chiral_count_cw <= max_chiral_cw:
            continue
            
        if chiral_count1_ccw == chiral_count2_ccw == 0:
            if chiral_count_ccw != 0:
                continue
        elif not min_chiral_ccw <= chiral_count_ccw <= max_chiral_ccw:
            continue

        filtered_list.append(mol)

    return filtered_list