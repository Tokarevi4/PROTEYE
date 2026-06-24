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
    idx = torch.tensor(
        [AA_TO_INDEX.get(r, -1) for r in residues],
        dtype=torch.long
    )

    features = torch.zeros((len(residues), len(AMINO_ACIDS) + 1))
    
    for i, r in enumerate(residues):
        if r in AA_TO_INDEX:
            features[i, AA_TO_INDEX[r]] = 1
        else:
            features[i, -1] = 1   # UNKNOWN

    valid = idx >= 0
    features[valid, idx[valid]] = 1.0

    return features