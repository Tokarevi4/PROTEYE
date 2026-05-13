# ProtEye

**Generative modeling of protein conformational states using geometric deep learning and diffusion-inspired neural architectures.**

Proteins are not static objects.  Traditional structural biology often represents proteins as fixed conformations, while real proteins continuously transition between multiple structural states.  ProtEye models this conformational variability using AI-driven generative approaches that learn from PDB data and produce plausible alternative 3D conformations.

---

## Overview

| Component | Description |
|-----------|-------------|
| `proTeye/data/` | PDB parsing, kNN graph construction, PyTorch dataset |
| `proTeye/models/` | GNN encoder, DDPM diffusion model, full conformer generator |
| `proTeye/training/` | Training loop, loss functions |
| `proTeye/utils/` | Dihedral angles, local frames, RMSD, Kabsch alignment |
| `scripts/` | `train.py` and `generate.py` CLI entry points |
| `tests/` | Unit tests for all modules |

### Architecture

```
Input PDB
    │
    ▼
PDB Loader ──▶ ProteinStructure (backbone coords + residue types)
    │
    ▼
Graph Builder ──▶ ProteinGraph (kNN edges, node/edge features)
    │
    ▼
GNN Encoder ──▶ per-residue embeddings (SE(3)-invariant)
    │
    ▼
DDPM Diffusion ──▶ alternative Cα coordinate sets
    │
    ▼
Generated conformations (saved as PDB)
```

---

## Installation

```bash
git clone https://github.com/Tokarevi4/ProtEye.git
cd ProtEye
pip install -e .
```

Dependencies: PyTorch ≥ 2.0, BioPython ≥ 1.81, NumPy, SciPy, tqdm.

---

## Quick Start

### Training

```bash
python scripts/train.py \
    --pdb_dir ./data/pdb_files \
    --output_dir ./checkpoints \
    --epochs 100 \
    --hidden_dim 128 \
    --diffusion_steps 200
```

### Generating conformations

```bash
python scripts/generate.py \
    --pdb ./input.pdb \
    --checkpoint ./checkpoints/model_final.pt \
    --num_samples 10 \
    --output_dir ./generated
```

Each generated conformation is saved as a Cα-only PDB file that can be opened in PyMOL, ChimeraX, or any molecular viewer.

---

## Python API

```python
from proTeye.data.pdb_loader import PDBLoader
from proTeye.data.graph_builder import ProteinGraphBuilder
from proTeye.models.conformer import ConformerGenerator

# Load a protein structure
loader = PDBLoader()
structures = loader.load("input.pdb")
protein = structures[0]

# Build the graph representation
graph = ProteinGraphBuilder(k=10).build(protein)

# Build and run the model
model = ConformerGenerator(
    node_input_dim=graph.node_features.shape[-1],
    edge_input_dim=graph.edge_features.shape[-1],
)
conformations = model.generate(graph, num_samples=5)
# conformations: list of (N, 3) tensors with Cα coordinates
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Method Details

### Graph construction

Each protein is represented as a directed graph where nodes are residues and edges connect each residue to its *k* nearest neighbours in Cα space (default k = 10, max distance 15 Å).

**Node features** (dim = 24):
- Amino-acid one-hot encoding (21-dim)
- Normalised sequence position (1-dim)
- Virtual Cα–Cα bond lengths to adjacent residues (2-dim)

**Edge features** (dim = 5):
- Distance normalised by max distance
- Sequence separation
- Unit displacement vector (3-dim)

### GNN Encoder

A stack of message-passing layers, each computing:

```
m_ij = MLP([h_i ‖ h_j ‖ e_ij])
h_i' = LayerNorm(h_i + MLP([h_i ‖ mean_j(m_ij)]))
```

### Diffusion model

Standard DDPM with a linear noise schedule applied to Cα coordinates.  The denoising network is another GNN conditioned on per-residue embeddings and the sinusoidal time step embedding.  Training minimises the simplified objective:

```
L = E_{t,ε} ‖ε_θ(x_t, t, h) − ε‖²
```

---

## License

MIT — see [LICENSE](LICENSE).
