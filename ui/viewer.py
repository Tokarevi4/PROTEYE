import py3Dmol
from stmol import showmol


def show_pdb(pdb_path, height=500):

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(
        width=800,
        height=height
    )

    view.addModel(
        pdb_data,
        "pdb"
    )

    view.setStyle(
        {"cartoon": {}}
    )

    view.zoomTo()

    showmol(
        view,
        height=height,
        width=800
    )