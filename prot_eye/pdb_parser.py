from Bio.PDB import PDBParser
import numpy as np


def extract_ca_atoms(pdb_path):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "protein",
        pdb_path
    )

    ca_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:

                if "CA" in residue:

                    atom = residue["CA"]

                    ca_atoms.append({
                        "residue": residue.get_resname(),
                        "coord": atom.get_coord()
                    })

    return ca_atoms


def pdb_to_graph(pdb_path, cutoff=10.0):
    """
    Convert protein structure into graph.

    Returns:
        node_features : [N,3]
        edge_index : [2,E]
    """

    atoms = extract_ca_atoms(pdb_path)

    coords = np.array([
        atom["coord"]
        for atom in atoms
    ])

    n = len(coords)

    edges = []

    for i in range(n):
        for j in range(i + 1, n):

            dist = np.linalg.norm(
                coords[i] - coords[j]
            )

            if dist < cutoff:

                edges.append([i, j])
                edges.append([j, i])

    edge_index = np.array(edges).T

    return coords, edge_index


if __name__ == "__main__":

    coords, edge_index = pdb_to_graph(
        "data/sample/1AAR.pdb"
    )

    print("Nodes:", coords.shape)
    print("Edges:", edge_index.shape)