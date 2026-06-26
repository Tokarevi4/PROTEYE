import torch


def calculate_rmsd(
    coords_true,
    coords_pred
):

    diff = coords_true - coords_pred

    rmsd = torch.sqrt(
        torch.mean(
            torch.sum(
                diff ** 2,
                dim=1
            )
        )
    )

    return rmsd.item()