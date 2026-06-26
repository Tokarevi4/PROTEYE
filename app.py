import os
import streamlit as st
import streamlit.components.v1 as components
import torch
import py3Dmol

from inference.predictor import ProtEyePredictor
from prot_eye.spatial_tensor_builder import build_spatial_graph_tensors
from prot_eye.pdb_writer import write_predicted_structure
from metrics.rmsd import calculate_rmsd

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="ProtEye",
    layout="wide"
)

st.title("🧬 ProtEye")

st.markdown(
    """
    Protein Structure Reconstruction using
    E(n)-Equivariant Graph Neural Networks
    """
)


# ==================================================
# MODEL LOAD
# ==================================================
@st.cache_resource
def load_predictor():
    return ProtEyePredictor()


predictor = load_predictor()


# ==================================================
# PY3DMOL VIEWERS
# ==================================================
def show_pdb(pdb_path, height=500):
    with open(pdb_path, "r") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=700, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()

    components.html(view._make_html(), height=height, width=700)


def show_overlay(original_pdb, predicted_pdb, height=500):
    with open(original_pdb) as f:
        original = f.read()

    with open(predicted_pdb) as f:
        predicted = f.read()

    view = py3Dmol.view(width=900, height=height)

    # Original (Green)
    view.addModel(original, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "green"}})

    # Predicted (Red)
    view.addModel(predicted, "pdb")
    view.setStyle({"model": 1}, {"cartoon": {"color": "red"}})

    view.zoomTo()
    components.html(view._make_html(), height=height, width=900)


# ==================================================
# FILE UPLOAD & INTERACTION
# ==================================================
uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    input_path = "temp/input.pdb"
    output_path = "temp/predicted.pdb"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDB uploaded successfully.")

    # ДОБАВЛЕНО: Ползунок для управления уровнем шума (Noise STD)
    # По умолчанию стоит 0.5 — верхний предел обучения вашей модели
    noise_std = st.slider(
        "Select Synthetic Noise Level (Å)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )

    st.subheader("Original Clean Structure")
    show_pdb(input_path)

    if st.button("Run Reconstruction"):

        # ==========================================
        # GRAPH BUILD & SYNTHETIC NOISE
        # ==========================================
        with st.spinner("Building spatial graph from PDB..."):
            tensors = build_spatial_graph_tensors(input_path)

        # Сохраняем исходные чистые координаты как эталон для подсчета RMSD
        clean_target_coords = tensors["coords"].clone()

        # Накладываем синтетический шум на координаты, если ползунок > 0
        if noise_std > 0.0:
            noise_vectors = torch.randn_like(tensors["coords"])
            tensors["coords"] = tensors["coords"] + noise_vectors * noise_std

        residues = tensors["coords"].shape[0]
        edges = tensors["edge_index"].shape[1]

        c1, c2 = st.columns(2)
        c1.metric("Residues", int(residues))
        c2.metric("Edges", int(edges))

        # ==========================================
        # MODEL INFERENCE
        # ==========================================
        with st.spinner("Running ProtEye EGNN Denoising..."):
            # Модель принимает ЗАШУМЛЕННЫЕ координаты и возвращает очищенные
            pred_coords = predictor.predict(tensors)

        # ==========================================
        # METRICS CALCULATION (CORRECTED)
        # ==========================================
        # 1. Ошибка структуры ДО модели (величина наложенного шума)
        initial_rmsd = calculate_rmsd(tensors["coords"], clean_target_coords)

        # 2. Ошибка структуры ПОСЛЕ модели (качество восстановления)
        final_rmsd = calculate_rmsd(pred_coords, clean_target_coords)

        # 3. Физическая амплитуда движения атомов, совершенная моделью
        shift = torch.norm(pred_coords - tensors["coords"], dim=1)
        mean_shift = shift.mean().item()
        max_shift = shift.max().item()

        # Вывод трех ключевых метрик на дашборд
        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Noise RMSD (Å)", f"{initial_rmsd:.4f}")
        c2.metric("Final Cleansed RMSD (Å)", f"{final_rmsd:.4f}")
        c3.metric("Mean Atom Shift (Å)", f"{mean_shift:.4f}")

        # Дополнительно выводим максимальный сдвиг в лог
        st.info(f"💡 Max Coordinate Correction: {max_shift:.4f} Å")

        # ==========================================
        # SAVE RECONSTRUCTED PDB
        # ==========================================
        write_predicted_structure(
            input_path,
            output_path,
            pred_coords.numpy()
        )

        st.success("Reconstruction completed!")

        # ==========================================
        # SIDE-BY-SIDE VIEW
        # ==========================================
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (Clean)")
            show_pdb(input_path)
        with col2:
            st.subheader("ProtEye Predicted")
            show_pdb(output_path)

        # ==========================================
        # OVERLAY COMPARISON
        # ==========================================
        st.subheader("Overlay Comparison")
        st.caption("Green = Original Ideal | Red = ProtEye Reconstructed")
        show_overlay(input_path, output_path)

        # ==========================================
        # DOWNLOAD BUTTON
        # ==========================================
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Predicted PDB",
                data=f,
                file_name="predicted.pdb",
                mime="chemical/x-pdb"
            )
