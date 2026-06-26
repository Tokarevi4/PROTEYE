from Bio.PDB import PDBParser, PDBIO


def write_predicted_structure(
    input_pdb,
    output_pdb,
    new_coords
):
    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "protein",
        input_pdb
    )

    idx = 0

    for model in structure:
        for chain in model:
            for residue in chain:

                if "CA" in residue:

                    residue["CA"].set_coord(
                        new_coords[idx]
                    )

                    idx += 1

    io = PDBIO()

    io.set_structure(structure)

    io.save(output_pdb)