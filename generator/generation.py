from random import randrange
import numpy as np 
import random
import time
import rdkit 

from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol, MolToSmiles as mol2smi, AllChem, rdMolAlign, Descriptors, Draw
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import rdFMCS, MolFromSmarts
from selfies import encoder, decoder

#from openeye import oechem, oeomega, oeshape 
#from openeye.oeomega import OEOmega
#from openeye.oechem import OEMol, OEParseSmiles, OEMolToSmiles

import selfies
from . import fingerprint
from . import filters
from . import scoring


def sanitize_smiles(smi):
    '''Return a canonical SMILES representation of smi
    
    Parameters:
    smi (string) : SMILES string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit Mol object                          (None if invalid SMILES string smi)
    smi_canon (string)          : Canonicalized SMILES representation of smi (None if invalid SMILES string smi)
    conversion_successful (bool): True/False to indicate if conversion was successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=True, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def get_selfie_chars(selfie):
    '''Obtain a list of all SELFIE characters in string selfie
    
    Parameters: 
    selfie (string) : A SELFIE string representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: List of SELFIE characters present in the molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE symbols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie


def has_stereochemistry(mol):
    '''Check if the molecule has any specified stereochemistry.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RDKit Mol object
    
    Returns:
    has_stereo (bool): True if the molecule has stereochemistry, False otherwise
    '''
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            return True
    return False


def molecule_to_selfies(mol):
    '''Convert an RDKit Mol object to SELFIES notation while preserving chirality.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RDKit Mol object
    
    Returns:
    selfie (string) : SELFIES notation of the molecule
    '''
    if has_stereochemistry(mol):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return encoder(Chem.MolToSmiles(mol))


def selfies_to_molecule(selfie):
    '''Convert a SELFIES notation to an RDKit Mol object.
    
    Parameters:
    selfie (string) : SELFIES notation
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RDKit Mol object
    '''
    smiles = decoder(selfie)
    mol, _, _ = sanitize_smiles(smiles)
    return mol


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string (only one mutation on selfie is performed)
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 33.3% probability, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character
    
    Parameters:
    selfie (string)  : SELFIE string to be mutated 
    max_molecules_len (int)  : Mutations of SELFIE string are allowed up to this length
    write_fail_cases (bool)  : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated (string)  : Mutated SELFIE string
    smiles_canon (string)  : Canonical SMILES of mutated SELFIE string
    '''
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters 

        choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
#        choice_ls = [2, 3] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]
        
        # Insert a character in a Random Location
        if random_choice == 1: 
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]
            
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        elif random_choice == 2:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index + 1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index + 1:]
                
        # Delete a random character
        elif random_choice == 3: 
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index + 1:]
                
        else: 
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
                if has_stereochemistry(mol):
                    # Preserve chirality information
                    selfie_mutated = molecule_to_selfies(mol)
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: ' + str(sf) + ' To Obtain: ' + str(selfie_mutated) + '\n')
                f.close()
    
    return selfie_mutated, smiles_canon


def get_mutated_SELFIES(selfies_ls, num_mutations): 
    ''' Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times. 
    
    Parameters:
    selfies_ls (list)  : A list of SELFIES 
    num_mutations (int): Number of mutations to perform on each SELFIES within 'selfies_ls'
    
    Returns:
    selfies_ls (list)  : A list of mutated SELFIES
    
    '''
    for _ in range(num_mutations): 
        selfie_ls_mut_ls = []
        for str_ in selfies_ls: 
            
            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations
            
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)
        
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls


def randomize_smiles(mol):
    '''Returns a random (dearomatized) SMILES given an RDKit Mol object of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RDKit Mol object
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RDKit Mol object
    '''
    if not mol:
        return None

    Chem.Kekulize(mol)
        
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True, kekuleSmiles=True)


def get_random_smiles(smi, num_random_samples): 
    '''Obtain 'num_random_samples' non-unique SMILES orderings of smi
    
    Parameters:
    smi (string)            : Input SMILES string (needs to be a valid molecule)
    num_random_samples (int): Number of unique different SMILES orderings to form 
    
    Returns:
    randomized_smile_orderings (list) : List of SMILES strings
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        raise Exception('Invalid starting structure encountered')
    randomized_smile_orderings = [randomize_smiles(mol) for _ in range(num_random_samples)]
    randomized_smile_orderings = list(set(randomized_smile_orderings)) # Only consider unique SMILES strings
    return randomized_smile_orderings


def obtain_path(starting_smile, target_smile, filter_path=True, fp_type="ECFP4"): 
    '''Obtain a path/chemical path from starting_smile to target_smile
    
    Parameters:
    starting_smile (string) : SMILES string (needs to be a valid molecule)
    target_smile (int)      : SMILES string (needs to be a valid molecule)
    filter_path (bool)      : If True, a chemical path is returned, else only a path
    
    Returns:
    path_smiles (list)                  : List of SMILES strings in path between starting_smile & target_smile
    path_fp_scores (list of floats)     : Fingerprint similarity to 'target_smile' for each SMILES in path_smiles
    smiles_path (list)                  : List of SMILES strings in CHEMICAL path between starting_smile & target_smile (if filter_path==False, then empty)
    filtered_path_score (list of floats): Fingerprint similarity to 'target_smile' for each SMILES in smiles_path (if filter_path==False, then empty)
    '''
    starting_selfie = molecule_to_selfies(Chem.MolFromSmiles(starting_smile))
    target_selfie = molecule_to_selfies(Chem.MolFromSmiles(target_smile))
    
    starting_selfie_chars = get_selfie_chars(starting_selfie)
    target_selfie_chars = get_selfie_chars(target_selfie)
    
    # Pad the smaller string
    if len(starting_selfie_chars) < len(target_selfie_chars): 
        starting_selfie_chars += [' '] * (len(target_selfie_chars) - len(starting_selfie_chars))
    else: 
        target_selfie_chars += [' '] * (len(starting_selfie_chars) - len(target_selfie_chars))
    
    indices_diff = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != target_selfie_chars[i]]
    path = {}
    path[0] = starting_selfie_chars
    
    for iter_ in range(len(indices_diff)): 
        idx = np.random.choice(indices_diff, 1)[0] # Index to be operated on
        indices_diff.remove(idx)                   # Remove that index
        
        # Select the last member of path: 
        path_member = path[iter_].copy()
        
        # Mutate that character to the correct value: 
        path_member[idx] = target_selfie_chars[idx]
        path[iter_ + 1] = path_member.copy()
    
    # Collapse path to make them into SELFIE strings
    paths_selfies = []
    for i in range(len(path)):
        selfie_str = ''.join(x for x in path[i])
        paths_selfies.append(selfie_str.replace(' ', ''))
        
    if paths_selfies[-1] != target_selfie: 
        raise Exception("Unable to discover target structure!")
    
    # Obtain similarity scores and only choose the increasing members
    path_smiles = [decoder(x) for x in paths_selfies]
    path_fp_scores = []
    filtered_path_score = []
    smiles_path = []

    if filter_path:
        path_fp_scores = fingerprint.get_fp_scores(path_smiles, target_smile, fp_type)

        filtered_path_score = []
        smiles_path = []
        for i in range(1, len(path_fp_scores) - 1): 
            if i == 1: 
                filtered_path_score.append(path_fp_scores[1])
                smiles_path.append(path_smiles[i])
                continue
            if filtered_path_score[-1] < path_fp_scores[i]:
                filtered_path_score.append(path_fp_scores[i])
                smiles_path.append(path_smiles[i])

    return path_smiles, path_fp_scores, smiles_path, filtered_path_score


def get_compr_paths(starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional, fp_type):
    '''Obtain multiple paths/chemical paths from starting_smile to target_smile. 
    
    Parameters:
    starting_smile (string)     : SMILES string (needs to be a valid molecule)
    target_smile (int)          : SMILES string (needs to be a valid molecule)
    num_tries (int)             : Number of path/chemical path attempts between the exact same smiles
    num_random_samples (int)    : Number of different SMILES string orderings to consider for starting_smile & target_smile 
    collect_bidirectional (bool): If True, forms paths from target_smiles-> target_smiles (doubles the number of paths)
    fp_type (string)            : Type of fingerprint  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6) 

    Returns:
    smiles_paths_dir1 (list): List of paths containing SMILES in the path between starting_smile -> target_smile
    smiles_paths_dir2 (list): List of paths containing SMILES in the path between target_smile -> starting_smile
    '''
    starting_smile_rand_ord = get_random_smiles(starting_smile, num_random_samples=num_random_samples)
    target_smile_rand_ord = get_random_smiles(target_smile, num_random_samples=num_random_samples)
    
    smiles_paths_dir1 = [] # All paths from starting_smile -> target_smile
    for smi_start in starting_smile_rand_ord: 
        for smi_target in target_smile_rand_ord: 
            
            if Chem.MolFromSmiles(smi_start) is None or Chem.MolFromSmiles(smi_target) is None: 
                raise Exception('Invalid structures')
                
            for _ in range(num_tries):
                path, _, _, _ = obtain_path(smi_start, smi_target, filter_path=True, fp_type=fp_type)
                smiles_paths_dir1.append(path)
    
    smiles_paths_dir2 = [] # All paths from starting_smile -> target_smile
    if collect_bidirectional: 
        starting_smile_rand_ord = get_random_smiles(target_smile, num_random_samples=num_random_samples)
        target_smile_rand_ord = get_random_smiles(starting_smile, num_random_samples=num_random_samples)
        
        for smi_start in starting_smile_rand_ord: 
            for smi_target in target_smile_rand_ord: 
                
                if Chem.MolFromSmiles(smi_start) is None or Chem.MolFromSmiles(smi_target) is None: 
                    raise Exception('Invalid structures')
        
            for _ in range(num_tries):
                path, _, _, _ = obtain_path(smi_start, smi_target, filter_path=True)
                smiles_paths_dir2.append(path)
                    
    return smiles_paths_dir1, smiles_paths_dir2


def generation_path(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, fp_type):
    """
    Runs the path-based generation and subsequently scores the generated molecules, returning the best predicted molecule.
    
    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        num_tries (int): The number of attempts for the path generation process.
        num_random_smiles (int): The number of random SMILES to be generated.
        collect_bidirectional (bool): Flag to determine if bidirectional path generation is considered.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        str: The SMILES representation of the best predicted intermediate molecule.
    """
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    
    # generate path
    generated_paths = get_compr_paths(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, fp_type=fp_type)
    
    # flatten lists
    smiles_generated_paths = filters.flatten(generated_paths)
    print('total number of path based generated intermediates: ', len(smiles_generated_paths))

    # write smiles to mols
    mols_generated_paths = [Chem.MolFromSmiles(smi) for smi in smiles_generated_paths]

    # drop duplicates
    uniq = filters.drop_duplicates(liga_smiles, ligb_smiles, smiles_generated_paths)
    
    # apply filters path based generation
    filtered_smiles_path = filters.filters_path_based_generation(liga_smiles, ligb_smiles, uniq)
    print('total number of path based generated intermediates after filters: ', len(filtered_smiles_path))

    # scoring path based generation
    subset_tanimoto, best_scores = scoring.score_median_mols(liga_smiles, ligb_smiles, filtered_smiles_path, exponent_path)
    print(subset_tanimoto, best_scores)

    # drop duplicates 
    #subset_tanimoto = filters.drop_duplicates(liga_smiles, ligb_smiles, smiles_)

    # If subset_tanimoto is empty, return None
    if not subset_tanimoto:
        print("subset_tanimoto is empty. Skipping this iteration.")
        return None
    
    # select path based intermediate
    intermed_smiles = subset_tanimoto[0]
    
    return intermed_smiles


def generate_multiple_paths(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, n_rounds, fp_type):
    """
    Runs the path-based generation n amount of times, finding multiple intermediates and so increasing the size of the searched chemical space.
    
    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        num_tries (int): The number of attempts for the path generation process.
        num_random_smiles (int): The number of random SMILES to be generated.
        collect_bidirectional (bool): Flag to determine if bidirectional path generation is considered.
        n_rounds (int): The number of rounds to perform the path-based generation.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        list: The list of unique top-scoring intermediates.
    """    
    # create list for multiple intermed_smiles
    intermed_smiles_list = []
    for _ in range(n_rounds):
        top_intermediate = generation_path(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, fp_type=fp_type)
        intermed_smiles_list.append(top_intermediate)
        
    intermed_smiles = filters.drop_duplicates(liga_smiles, ligb_smiles, intermed_smiles_list)
    print('Number of unique top1-scoring intermediates: ',len(intermed_smiles))
    
    return intermed_smiles


def generate_local_chemical_space(intermed_smiles, num_random_samples, num_mutation_ls, fp_type):
    """
    Generates the local chemical space around a molecule.

    Args:
        intermed_smiles (str): The SMILES representation of the intermediate molecule.
        num_random_samples (int): The number of random samples for the local chemical space generation.
        num_mutation_ls (list): List containing the number of mutations to be performed.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        list: The list of unique molecules in the generated local chemical space including the intermediate molecule.
    """    
    intermed_mol = Chem.MolFromSmiles(intermed_smiles)
    
    total_time = time.time()
    #num_random_samples = 50000 # For a more exhaustive search! 
    #num_random_samples = num_random_samples     
    #num_mutation_ls    = [1,2]

    smi=intermed_smiles
    mol = intermed_mol
    if mol == None: 
        raise Exception('Invalid starting structure encountered')

    start_time = time.time()
    randomized_smile_orderings  = [randomize_smiles(mol) for _ in range(num_random_samples)]

    # Convert all the molecules to SELFIES
    selfies_ls = [encoder(x) for x in randomized_smile_orderings]
    print('Randomized molecules (in SELFIES) time: ', time.time()-start_time)


    all_smiles_collect = []
    all_smiles_collect_broken = []

    start_time = time.time()
    for num_mutations in num_mutation_ls: 
        # Mutate the SELFIES: 
        selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)

        # Convert back to SMILES: 
        smiles_back = [decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_smiles_collect_broken.append(smiles_back)


    print('Mutation obtainment time (back to smiles): ', time.time()-start_time)


    # Work on:  all_smiles_collect
    start_time = time.time()
    canon_smi_ls = []
    for item in all_smiles_collect: 
        mol, smi_canon, did_convert = sanitize_smiles(item)
        if mol == None or smi_canon == '' or did_convert == False: 
            raise Exception('Invalid smile string found')
        canon_smi_ls.append(smi_canon)
    canon_smi_ls        = list(set(canon_smi_ls))
    print('Unique mutated structure obtainment time: ', time.time()-start_time)

    start_time = time.time()
    canon_smi_ls_scores = fingerprint.get_fp_scores(canon_smi_ls, target_smi=smi, fp_type=fp_type)
    print('Fingerprint calculation time: ', time.time()-start_time)
    print('Total time: ', time.time()-total_time)
    
    # add the intermediate mol to the list
    canon_smi_ls.append(intermed_smiles)
    
    return canon_smi_ls


def generate_chemical_space(liga_smiles, ligb_smiles, intermediate_smiles, num_random_samples, num_mutation_ls, fp_type):
    """
    Generates the chemical space between two ligands using multiple intermediate molecules.

    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        intermediate_smiles (list): List of SMILES representation of intermediate molecules.
        num_random_samples (int): The number of random samples for the local chemical space generation.
        num_mutation_ls (list): List containing the number of mutations to be performed.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        list: The list of unique molecules in the generated chemical space.
    """    
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    
    
    all_filtered_smiles = []

    for smiles in intermediate_smiles:

        # generate local chemical space
        generated_chemical_space = generate_local_chemical_space(smiles, num_random_samples, num_mutation_ls, fp_type)
        print('total number of generated local chemical space intermediates for smiles-pair: ', len(generated_chemical_space))

        # apply filters local chemical space generation
        filtered_smiles = filters.filters_local_chemical_space_generation(liga_smiles, ligb_smiles, generated_chemical_space)
        print('total number of generated local chemical space intermediates after filters for smiles-pair: ', len(filtered_smiles))
        print('')
        
        all_filtered_smiles.extend(filtered_smiles)

    combined_smiles = filters.drop_duplicates(liga, ligb, all_filtered_smiles)    
    print('total number of unique generated local chemical space intermediates: ', len(combined_smiles))
    print('these are the final smiles:', combined_smiles)

    
    return combined_smiles


def generation_path_rocs(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, omega, fp_type):
    """
    Runs the path-based generation and subsequently scores the generated molecules, returning the best predicted molecule.
    
    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        num_tries (int): The number of attempts for the path generation process.
        num_random_smiles (int): The number of random SMILES to be generated.
        collect_bidirectional (bool): Flag to determine if bidirectional path generation is considered.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        str: The SMILES representation of the best predicted intermediate molecule.
    """
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    
    # generate path
    generated_paths = get_compr_paths(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, fp_type=fp_type)
    
    # flatten lists
    smiles_generated_paths = filters.flatten(generated_paths)
    print('total number of path based generated intermediates: ', len(smiles_generated_paths))

    # write smiles to mols
    mols_generated_paths = [Chem.MolFromSmiles(smi) for smi in smiles_generated_paths]

    # drop duplicates
    uniq = filters.drop_duplicates(liga_smiles, ligb_smiles, smiles_generated_paths)
    
    # apply filters path based generation
    filtered_smiles_path = filters.filters_path_based_generation(liga_smiles, ligb_smiles, uniq)
    print('total number of path based generated intermediates after filters: ', len(filtered_smiles_path))

    # scoring path based generation
    #subset_tanimoto, best_scores = scoring.score_median_mols(liga_smiles, ligb_smiles, filtered_smiles_path)

    # add rocs based scoring function
#    subset_rocs, best_scores = scoring.rocs(liga_smiles, filtered_smiles_path, ligb_smiles, exponent_path)
    subset_rocs, best_scores = scoring.rocs(liga_smiles, filtered_smiles_path, ligb_smiles, 100, omega, exponent_path)
#    print(subset_rocs, best_scores, "subsetrocs")
    
    # drop duplicates 
    #subset_tanimoto = filters.drop_duplicates(liga_smiles, ligb_smiles, smiles_)

    # If subset_tanimoto is empty, return None
    if not subset_rocs:
        print("subset_rocs is empty. Skipping this iteration.")
        return None
    
    # select path based intermediate
    #print(subset_rocs[0], "best scoring molecule per iteration")
    intermed_smiles = subset_rocs[0]
    
    return intermed_smiles


def generate_multiple_paths_rocs(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, n_rounds, fp_type):
    """
    Runs the path-based generation n amount of times, finding multiple intermediates and so increasing the size of the searched chemical space.
    
    Args:
        liga_smiles (str): The SMILES representation of the first ligand.
        ligb_smiles (str): The SMILES representation of the second ligand.
        num_tries (int): The number of attempts for the path generation process.
        num_random_smiles (int): The number of random SMILES to be generated.
        collect_bidirectional (bool): Flag to determine if bidirectional path generation is considered.
        n_rounds (int): The number of rounds to perform the path-based generation.
        fp_type (str): The fingerprint type used in scoring molecules.

    Returns:
        list: The list of unique top-scoring intermediates.
    """    
    omega = scoring.initialize_omega()
    
    # create list for multiple intermed_smiles
    intermed_smiles_list = []
    for _ in range(n_rounds):
        top_intermediate = generation_path_rocs(liga_smiles, ligb_smiles, num_tries, num_random_smiles, collect_bidirectional, exponent_path, omega, fp_type=fp_type)
        intermed_smiles_list.append(top_intermediate)
        
    intermed_smiles = filters.drop_duplicates(liga_smiles, ligb_smiles, intermed_smiles_list)
    print('Number of unique top1-scoring intermediates: ',len(intermed_smiles))
    
    return intermed_smiles

