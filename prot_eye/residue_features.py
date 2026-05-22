import torch


AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP",
    "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
]


AA_TO_INDEX = {
    aa: i for i, aa in enumerate(AMINO_ACIDS)
}


def one_hot_encode_residues(residues):

    features = torch.zeros(
        (len(residues), len(AMINO_ACIDS)),
        dtype=torch.float32
    )

    for i, residue in enumerate(residues):

        if residue in AA_TO_INDEX:

            idx = AA_TO_INDEX[residue]

            features[i, idx] = 1.0

    return features