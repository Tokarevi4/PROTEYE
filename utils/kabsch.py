import torch


def kabsch_align(P, Q):
    """
    P - исходные координаты [N,3]
    Q - предсказанные координаты [N,3]

    Возвращает:
        Q_aligned
    """

    P_centroid = P.mean(dim=0)
    Q_centroid = Q.mean(dim=0)

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    H = Q_centered.T @ P_centered

    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T

    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    Q_rot = Q_centered @ R

    Q_aligned = Q_rot + P_centroid

    return Q_aligned