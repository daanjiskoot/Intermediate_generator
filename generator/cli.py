import argparse
from .intermediates import intermediates

def main():
    parser = argparse.ArgumentParser(description='Calculate intermediates for molecular transformations.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sdf_files', 'i1', nargs=2, type=str, help='Paths to two SDF files containing molecules.')
    group.add_argument('--mol2_files', 'i2', nargs=2, type=str, help='Paths to two mol2 files containing molecules.')
    group.add_argument('--text_file', 'i3', type=str, help='Path to a text file containing the perts_to_intrap list.')
    
    parser.add_argument('--base_dir', '-b', type=str, required=True, help='Location to save images of pairs with intermediates.')
    parser.add_argument('--lig_path', '-l', type=str, required=True, help='Path to your ligand folder')
    
    parser.add_argument('--num_tries', '-n', type=int, default=10, help='Number of path/chemical path attempts between the exact same smiles.')
    parser.add_argument('--num_random_smiles', '-r', type=int, default=10, help='Number of different SMILES string orderings to consider for starting_smile & target_smile.')
    parser.add_argument('--collect_bidirectional', '-c', action='store_true', default=True, help='Whether to collect bidirectional paths.')
    parser.add_argument('--fp_type', '-f', type=str, default="ECFP4", help='Fingerprint type to use.')
    parser.add_argument('--num_random_samples', '-S', type=int, default=500, help='Number of generated intermediates per mutation in the local chemical space generation.')
    parser.add_argument('--num_mutation_ls', '-m', type=int, nargs='+', default=[1, 2], help='Number of mutations per molecule in the local chemical space generation.')
    parser.add_argument('--n_rounds', '-R', type=int, default=5, help='Number of rounds in the path-based generation.')
    parser.add_argument('--exponent_path', '-e', type=int, default=4, help='Exponent path value.')
    parser.add_argument('--exponent_local_chemical_space', '-E', type=int, default=2, help='Exponent for local chemical space value.')
    parser.add_argument('--sdf', '-d', action='store_true', default=True, help='Output to SDF.')
    parser.add_argument('--svg', '-v', action='store_true', default=True, help='Output to SVG.')
    parser.add_argument('--png', '-p', action='store_true', default=False, help='Output to PNG.')
    parser.add_argument('--scoring_method', '-M', type=str, default="2D", help='Scoring method to use.')
    parser.add_argument('--contribution_lomap', '-L', type=float, default=0.2, help='Contribution of LoMAP.')
    parser.add_argument('--contribution_similarity', '-T', type=float, default=0.8, help='Contribution of similarity.')
    
    args = parser.parse_args()


    intermediates(
        args.sdf_files, 
        args.text_file, 
        args.base_dir, 
        args.num_tries, 
        args.num_random_smiles, 
        args.collect_bidirectional, 
        args.n_rounds, 
        args.fp_type, 
        args.num_random_samples, 
        args.num_mutation_ls,
        args.lig_path,
        args.exponent_path,
        args.exponent_local_chemical_space,
        args.sdf,
        args.svg,
        args.png,
        args.scoring_method,
        args.contribution_lomap,
        args.contribution_similarity
    )
    
if __name__ == "__main__":
    main()

