from pathlib import Path
import py3Dmol


BASE_DIR = Path(__file__).resolve().parent.parent

PDB_PATH = BASE_DIR / "data" / "sample" / "1AAR.pdb"


def visualize_pdb(pdb_path):

    with open(pdb_path, "r") as file:
        pdb_data = file.read()

    viewer = py3Dmol.view(
        width=900,
        height=700
    )

    viewer.addModel(
        pdb_data,
        "pdb"
    )

    viewer.setStyle({
        "cartoon": {
            "color": "spectrum"
        }
    })

    viewer.zoomTo()

    return viewer