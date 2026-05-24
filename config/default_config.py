from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {

    "pdb_path":
        BASE_DIR / "data" / "sample" / "1UBQ.pdb",

    "radius":
        8.0,

    "noise_std":
        1.0,

    "epochs":
        1500,

    "learning_rate":
        0.001,

    "output_dir":
        BASE_DIR / "outputs",

    "weights_dir":
        BASE_DIR / "weights"
}