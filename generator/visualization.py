import rdkit
import matplotlib.pyplot as plt
import time
import openbabel
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, rdMolAlign
from rdkit.Chem.Draw import MolToImage
import numpy as np
from scipy.spatial.distance import cdist

import filters

def draw_molecule_with_mcs(mol, mcs_mol, size=(400, 400)):
    """
    Draw a molecule with the maximum common substructure (MCS) highlighted.
    
    Args:
    mol (rdkit.Chem.rdchem.Mol): The molecule to draw.
    mcs_mol (rdkit.Chem.rdchem.Mol): The MCS molecule.
    size (tuple of int): Size of the image (width, height).
    
    Returns:
    PIL.PngImagePlugin.PngImageFile: An image of the molecule.
    """
    match = mol.GetSubstructMatch(mcs_mol)
    match_bonds = []

    for bond in mcs_mol.GetBonds():
        atom1 = match[bond.GetBeginAtomIdx()]
        atom2 = match[bond.GetEndAtomIdx()]
        match_bonds.append(mol.GetBondBetweenAtoms(atom1, atom2).GetIdx())

    return MolToImage(mol, size=size, highlightAtoms=match, highlightBonds=match_bonds)


def alignLigands(liga, intermediate, ligb):
    """
    Align a triplet of ligands to the first one for better visualization of differences.
    
    Args:
    liga, intermediate, ligb (rdkit.Chem.rdchem.Mol): The ligands to align.
    
    Returns:
    tuple of rdkit.Chem.rdchem.Mol: The aligned ligands.
    """

    mcs_result = rdFMCS.FindMCS([liga, intermediate, ligb],
                                 atomCompare=rdFMCS.AtomCompare.CompareAny,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 matchValences=False,
                                 ringMatchesRingOnly=True,
                                 completeRingsOnly=True,
                                 matchChiralTag=False,
                                 timeout=2)

    template = Chem.MolFromSmarts(mcs_result.smartsString)
    AllChem.Compute2DCoords(template)

    for lig in [liga, intermediate, ligb]:
        AllChem.GenerateDepictionMatching2DStructure(lig, template)
    
    return liga, intermediate, ligb


def visualise(liga_smiles, ligb_smiles, selected_intermediate_smiles, mins, secs, filepath, filename, svg, png):
    """
    Visualise and save an image of three aligned ligands with the Maximum Common Substructure (MCS) highlighted.
    
    Args:
        liga_smiles, ligb_smiles, selected_intermediate_smiles (str): The SMILES strings for the ligands to visualize.
        mins, secs (int): The minutes and seconds elapsed.
        filepath (str): The path to the location where the image is to be saved.
        filename (str): The name of the file.
    """
    # get intermediate
    liga = Chem.MolFromSmiles(liga_smiles)
    ligb = Chem.MolFromSmiles(ligb_smiles)
    intermediate = Chem.MolFromSmiles(selected_intermediate_smiles)

    # find MCS of the 3 molecules
    mcs_mol = filters.find_mcs_3_mols(liga_smiles, ligb_smiles, selected_intermediate_smiles)

    # Align ligands 
    molecules = alignLigands(liga, intermediate, ligb)

    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1]})

    # Generate the images with the MCS highlighted
    img0 = draw_molecule_with_mcs(liga, mcs_mol)
    img1 = draw_molecule_with_mcs(intermediate, mcs_mol)
    img2 = draw_molecule_with_mcs(ligb, mcs_mol)

    # Plot the images
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    axes[0].text(-0.05, 0.5, f'{filename}', rotation=90, va='center', fontsize=10, transform=axes[0].transAxes)


    # add elapsed time to the first image
    axes[0].text(0.25, 0.95, f'Elapsed Time: {int(mins)}m {int(secs)}s', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    # Turn off axes and set titles.
    titles = ["$\lambda = 0$", "Intermediate", "$\lambda = 1$"]
    for ax, title in zip(axes, titles):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title(title)

    # set spacing between plots
    plt.subplots_adjust(wspace=0.08)

    # options to save image
    
    # adjust save options if we use multiple input molecules
    if svg == True:
        plt.savefig(f'{filepath}.svg')  # For SVG format
    if png == True:
        plt.savefig(f'{filepath}.png')  # For PNG format

    plt.show()

def calculate_rmsd(ref_coords, prb_coords):
    return np.sqrt(np.mean(np.sum(np.square(ref_coords - prb_coords), axis=1)))

def align_intermediates_to_references(refmol1, refmol2, intermediate_smiles, outfile, n_conformers=1):
    """
    Align an intermediate molecule to two reference molecules separately. 
    The aligned molecules are saved in an sdf file.
    """
    mols = [refmol1, refmol2, Chem.MolFromSmiles(intermediate_smiles)]
    
    # Add hydrogens and generate conformers for the intermediate molecule
    mols[2] = Chem.AddHs(mols[2])
    AllChem.EmbedMultipleConfs(mols[2], numConfs=n_conformers, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    
    # Find the MCS:
    mcs = rdFMCS.FindMCS(mols, threshold=0.8, completeRingsOnly=True, ringMatchesRingOnly=True)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    
    writer = Chem.SDWriter(outfile)
    
    best_rmsd = [float('inf'), float('inf')]  # Initialize best RMSD as infinity for both reference molecules
    best_conformer = [None, None]  # Initialize best conformer as None for both reference molecules
    
    for conf_id in range(n_conformers):
        for idx, refmol in enumerate([refmol1, refmol2]):
            intermediate_copy = Chem.Mol(mols[2])  # Copy the intermediate molecule for alignment
            intermediate_copy.RemoveAllConformers()
            
            try:
                intermediate_copy.AddConformer(mols[2].GetConformer(conf_id), assignId=True)
            except ValueError:
                # If there's a ValueError (e.g., conformer ID doesn't exist), just continue
                continue

            refMatch = refmol.GetSubstructMatch(patt)
            smarts_mcs = intermediate_copy.GetSubstructMatch(patt)
            rdMolAlign.AlignMol(intermediate_copy, refmol, atomMap=list(zip(smarts_mcs, refMatch)))

            # Extract coordinates for matched atoms
            ref_conf = refmol.GetConformer()
            prb_conf = intermediate_copy.GetConformer()
            ref_coords = np.array([list(ref_conf.GetAtomPosition(i)) for i in refMatch])
            prb_coords = np.array([list(prb_conf.GetAtomPosition(i)) for i in smarts_mcs])

            # Calculate RMSD and keep track of the conformer with the lowest RMSD
            rmsd = calculate_rmsd(ref_coords, prb_coords)
            if rmsd < best_rmsd[idx]:
                best_rmsd[idx] = rmsd
                best_conformer[idx] = intermediate_copy

    # Write the conformers with the lowest RMSD for both refs as output
#    for conformer in best_conformer:
 #       if conformer is not None:
  #          writer.write(conformer)
            
    
    # only write one conformer to output file
    if best_conformer[0] is not None:
        writer.write(best_conformer[0])

    # Comment out or remove the next two lines to not write the conformer aligned to the second reference molecule
    #if best_conformer[1] is not None:
    #    writer.write(best_conformer[1])
    
    writer.close()

def mol2_to_rdkit(path):
    """Convert a mol2 file to an RDKit Mol object using Pybel."""
    with open(path, 'r') as f:
        mol2_text = f.read()
    mol = pybel.readstring("mol2", mol2_text)
    return Chem.MolFromMolBlock(mol.write("sdf"))

def get_mol_from_file(tgt, lig, file_extension, lig_path):
    path = lig_path
    if file_extension == ".sdf":
        return Chem.rdmolfiles.SDMolSupplier(path)[0]
    elif file_extension == ".mol2":
        return mol2_to_rdkit(path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
        
def fetch_molecule(target, ligand, lig_path, ext):
    for path_suffix in [ligand, f"lig_{ligand}"]:
        full_path = f"{lig_path}/{target}/{path_suffix}{ext}"
        try:
            return get_mol_from_file(target, ligand, ext, full_path)
        except OSError:
            continue
    return None