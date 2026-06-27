from pathlib import Path

raw = list(Path("data/raw_pdb").glob("*.pdb"))

print(f"Всего PDB: {len(raw)}")