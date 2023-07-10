import rdkit
import matplotlib.pyplot as plt
import time
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit.Chem.Draw import MolToImage

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


def visualise(liga_smiles, ligb_smiles, selected_intermediate_smiles, mins, secs, filepath, filename):
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
    
    plt.savefig(f'{filepath}.svg')  # For SVG n     format
    #plt.savefig('pert.png')  # For PNG format

    plt.show()

    