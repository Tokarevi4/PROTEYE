import numpy as np
from pathlib import Path

root = Path("data/train")

rmsds = []

for sample_dir in root.glob("sample_*"):

    x = np.load(sample_dir / "input.npy")
    y = np.load(sample_dir / "target.npy")

    rmsd = np.sqrt(
        np.mean(
            np.sum((x - y) ** 2, axis=1)
        )
    )

    rmsds.append(rmsd)

print("samples:", len(rmsds))
print("mean rmsd:", np.mean(rmsds))
print("min rmsd :", np.min(rmsds))
print("max rmsd :", np.max(rmsds))

print("input range:")
print(x.min(), x.max())

print("target range:")
print(y.min(), y.max())