import json
import numpy as np

protein = "11ND"

coords = np.load(f"data/processed/{protein}/coords.npy")
residues = np.load(f"data/processed/{protein}/residues.npy", allow_pickle=True)

with open(f"data/processed/{protein}/meta.json") as f:
    meta = json.load(f)

print("coords:", coords.shape, coords.dtype)
print("residues:", residues.shape, residues.dtype)
print("meta:", meta)


x = np.load("data/train/sample_002655/input.npy")
y = np.load("data/train/sample_002655/target.npy")

print("input:", x.shape, x.dtype)
print("target:", y.shape, y.dtype)