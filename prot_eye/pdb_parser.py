from Bio.PDB import PDBParser


def extract_ca_atoms(pdb_path):
    """
    Extract C-alpha atoms from PDB structure.
    """

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "protein",
        pdb_path
    )

    ca_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:

                if 'CA' in residue:

                    atom = residue['CA']

                    ca_atoms.append({
                        "residue": residue.get_resname(),
                        "coord": atom.get_coord()
                    })

    return ca_atoms


if __name__ == "__main__":

    atoms = extract_ca_atoms(
        "data/sample/1UBQ.pdb"
    )

    print(f"Extracted {len(atoms)} Cα atoms\n")

    for atom in atoms[:5]:
        print(atom)