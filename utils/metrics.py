import torch


def rmsd(P, Q):
    """
    P,Q: [N,3]
    """

    return torch.sqrt(
        torch.mean(
            torch.sum((P - Q) ** 2, dim=1)
        )
    )