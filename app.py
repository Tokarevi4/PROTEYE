import os
import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
from Bio.PDB import PDBParser
import py3Dmol
from streamlit_echarts import st_echarts

# Настройка глобальной темы отображения
st.set_page_config(
    page_title="ProtEye",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 ProtEye — EGNN Denoising")

# Импорты ваших внутренних модулей
from inference.predictor import ProtEyePredictor
from prot_eye.pdb_writer import write_predicted_structure
from utils.metrics import rmsd
from utils.kabsch import kabsch_align


st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) *,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) div {
        border-radius: 0px !important;
        border-color: transparent !important; 
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) {
        background-color: #000000 !important;
        padding: 12px !important;
        outline: 1px solid #30363d !important; 
        outline-offset: -1px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) iframe,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) [data-testid="stHorizontalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) [data-testid="element-container"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) [data-testid="stHtml"] {
        background-color: #000000 !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="viewer_box_style"]) div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        border: 1px solid #30363d !important;
        border-radius: 0px !important; /* Углы селектора тоже делаем строгими */
    }

    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    .sidebar-footer {
        margin-top: auto;
        padding-top: 20px;
        font-size: 0.8rem;
        color: #7d8597;
        text-align: center;
        border-top: 1px solid #2d3748;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# @st.cache_resource
def load_predictor():
    """
    Безопасно инициализирует ваш класс предиктора и кэширует его инстанс.
    """
    return ProtEyePredictor()

# Создаем глобальный объект предиктора
predictor = load_predictor()

def show_pdb_custom(pdb_path, style_opt="cartoon", height=400, width=540):
    with open(pdb_path, "r") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")
    view.setStyle({style_opt.lower(): {"color": "spectrum"}})
    
    view.setBackgroundColor('black')
    view.zoomTo()
    
    components.html(view._make_html(), height=height, width=width)


def show_overlay_custom(original_pdb, predicted_pdb, height=400, width=540):
    with open(original_pdb, "r") as f:
        original = f.read()

    with open(predicted_pdb, "r") as f:
        predicted = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(original, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "green"}})
    view.addModel(predicted, "pdb")
    view.setStyle({"model": 1}, {"cartoon": {"color": "red"}})

    view.setBackgroundColor('black')
    view.zoomTo()
    components.html(view._make_html(), height=height, width=width)


def calculate_all_dashboard_metrics(clean_target, noisy_coords, pred_coords):
    # 1. Выравнивание Кабша для зашумленного входа (ДО модели) для честного TM-score и RMSD
    noisy_aligned = kabsch_align(clean_target.cpu(), noisy_coords.cpu()).to(clean_target.device)

    # 2. Ошибка структуры ДО модели (величина реального наложенного шума после выравнивания)
    initial_rmsd = rmsd(noisy_aligned, clean_target)
    
    # 3. Выравнивание Кабша для результата работы модели (ПОСЛЕ денойзинга)
    pred_aligned = kabsch_align(clean_target.cpu(), pred_coords.cpu()).to(clean_target.device)
    
    # 4. Финальный RMSD после пространственного совмещения
    final_rmsd = rmsd(pred_aligned, clean_target)
    
    # 5. Физическая амплитуда движения атомов, совершенная моделью
    shift = torch.norm(pred_coords - noisy_coords, dim=1)
    mean_shift = shift.mean().item()
    max_shift = shift.max().item()
    
    # 6. Аппроксимация TM-score по стандартной формуле Zhang & Skolnick на базе ВЫРОВНЕННЫХ расстояний
    L = clean_target.shape[0]
    d0 = max(1.24 * np.power(max(L - 15, 0.5), 1/3) - 1.8, 0.5)
    
    # ИСПРАВЛЕНО: Считаем расстояния строго по пространственно совмещенным координатам
    dist_sq_before = torch.sum((clean_target - noisy_aligned) ** 2, dim=1)
    dist_sq_after = torch.sum((clean_target - pred_aligned) ** 2, dim=1)
    
    tm_before = torch.mean(1.0 / (1.0 + (dist_sq_before / (d0 ** 2)))).item()
    tm_after = torch.mean(1.0 / (1.0 + (dist_sq_after / (d0 ** 2)))).item()
    
    return {
        "rmsd_before": f"{initial_rmsd:.4f} Å",
        "rmsd_after": f"{final_rmsd:.4f} Å",
        "mean_shift": f"{mean_shift:.4f} Å",
        "max_shift": f"{max_shift:.4f} Å",
        "tm_before": f"{tm_before:.4f}",
        "tm_after": f"{tm_after:.4f}"
    }


with st.sidebar:
    st.title("🧬 ProtEye")
    st.caption("Protein Structure Denoising with EGNN")
    st.markdown("---")

    # 1. Загрузчик файла
    st.subheader("1. Загрузите PDB файл")
    uploaded_file = st.file_uploader(
        "Перетащите файл в это поле или нажмите для выбора:",
        type=["pdb"],
        label_visibility="collapsed"
    )

    # 2. Ползунок уровня искусственного шума (Noise STD)
    st.subheader("2. Уровень искусственного шума")
    noise_std = st.slider(
        "Интенсивность деформации (Å)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

    # Кнопки отправки и сброса формы
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_prediction = st.button("Запустить реконструкцию", type="primary", use_container_width=True)
    with col_btn2:
        reset_app = st.button("Сбросить", type="secondary", use_container_width=True)

    st.markdown("---")
    st.subheader("Информация о белке")
    info_container = st.container()

    # Пути сохранения временных файлов
    input_path = "temp/input.pdb"
    output_path = "temp/predicted.pdb"
    
    # Объявляем переменные и делаем их глобальными для доступа в главной области
    global pdb_text_content
    tensors = None
    clean_target_coords = None
    residues_count = 0
    edges_count = 0

    if uploaded_file:
        os.makedirs("temp", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdb_text_content = uploaded_file.getvalue().decode("utf-8")

        # Сборщик пространственного графа
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", input_path)
        clean_coords_list = [atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == "CA"]
        
        if len(clean_coords_list) > 0:
            clean_target_coords = torch.tensor(np.array(clean_coords_list, dtype=np.float32))
            residues_count = clean_target_coords.size(0)
            
            # ИСПРАВЛЕНО: Извлекаем трехбуквенные имена остатков для One-Hot эмбеддингов в модели
            residues_list = [res.get_resname().upper() for res in structure.get_residues()]
            # Синхронизируем длину массива аминокислот с количеством C-alpha атомов
            residues_list = residues_list[:residues_count]
            
            # Наложение синтетического шума на координаты атомов
            if noise_std > 0.0:
                noise_vectors = torch.randn_like(clean_target_coords) * noise_std
                noisy_coords = clean_target_coords + noise_vectors
            else:
                noisy_coords = clean_target_coords.clone()
                
            # ИСПРАВЛЕНО: Алгоритмическое построение ребер графа по радиусу отсечки 10.0 Å, как на обучении
            if residues_count > 1:
                dist_matrix = torch.cdist(noisy_coords, noisy_coords, p=2)
                # Матрица смежности: расстояние меньше 10.0 Ангстрем и исключаем главную диагональ (self-loops)
                adj_matrix = (dist_matrix < 10.0) & (~torch.eye(residues_count, dtype=torch.bool, device=clean_target_coords.device))
                edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            edges_count = edge_index.size(1)
            
            # ИСПРАВЛЕНО: Передаем в предиктор полный комплект данных (координаты, ребра, аминокислоты и уровень шума)
            tensors = {
                "coords": noisy_coords,
                "edge_index": edge_index,
                "residues": residues_list,
                "noise_std": noise_std
            }
            
            with info_container:
                st.markdown("📋 **Параметры графа молекулы:**")
                st.text(f"Остатков (узлов): {residues_count}")
                st.text(f"Рёбер (связей): {edges_count}")
                st.text("Тип графа: Radius-based Cutoff")  # Обновили для соответствия дипломной документации
                st.text("Радиус связи: 10.0 Å")            # Обновили для соответствия дипломной документации
        else:
            st.error("В PDB файле не найдены C-alpha атомы (CA)!")
    else:
        # Если файл не загружен, инициализируем пустой строкой, чтобы избежать NameError
        pdb_text_content = ""
        with info_container:
            st.info("Загрузите PDB файл, чтобы отобразить геометрические параметры графа молекулы.")

    st.markdown("<div class='sidebar-footer'>ProtEye EGNN v0.5</div>", unsafe_allow_html=True)

if uploaded_file and tensors is not None:
    
    is_reconstructed = False
    dashboard_metrics = None
    
    if run_prediction and predictor is not None:
        with st.spinner("Running ProtEye EGNN Denoising..."):
            pred_coords = predictor.predict(tensors)
            dashboard_metrics = calculate_all_dashboard_metrics(
                clean_target=clean_target_coords,
                noisy_coords=tensors["coords"],
                pred_coords=pred_coords
            )
            write_predicted_structure(input_path, output_path, pred_coords.numpy())
            is_reconstructed = True

    # Заголовок страницы и статус
    head_col1, head_col2 = st.columns(2)
    with head_col1:
        st.subheader("Визуализация и анализ")
    with head_col2:
        if is_reconstructed:
            st.markdown(
                """
                <div style='background-color: #065f46; color: #34d399; padding: 6px 12px; 
                border-radius: 20px; text-align: center; font-weight: 500; font-size: 0.85rem;
                border: 1px solid #059669; margin-top: 5px; width: 220px; float: right;'>
                    🟢 Реконструкция завершена
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Разделение экрана 50/50 на Левое и Правое окно
    vis_col1, vis_col2 = st.columns(2)
    
    # --- ЛЕВОЕ ОКНО: 3D структура молекулы (AMOLED БОКС) ---
    with vis_col1:
        with st.container(border=True):
            # Невидимый CSS-маркер для точечной покраски этой рамки
            st.markdown("<div class='viewer_box_style'></div>", unsafe_allow_html=True)
            
            # Элементы управления внутри бокса
            v_sub1, v_sub2 = st.columns(2)
            with v_sub1:
                st.markdown("**3D структура (PyMOL)** 🔗")
            with v_sub2:
                style_option = st.selectbox(
                    "Стиль:", ["Cartoon", "Stick", "Sphere"], 
                    label_visibility="collapsed", index=0, key="viewer_style_opt"
                )
            
            current_pdb_to_show = output_path if is_reconstructed else input_path
            show_pdb_custom(current_pdb_to_show, style_opt=style_option, width=540, height=400)

    # --- ПРАВОЕ ОКНО: Возвращаем 3D-граф на чисто чёрном фоне ---
    with vis_col2:
        with st.container(border=True):
            # Невидимый маркер для точечной подгонки
            st.markdown("<div class='viewer_box_style'></div>", unsafe_allow_html=True)
            
            g_sub1, g_sub2 = st.columns(2)
            with g_sub1:
                st.markdown("**3D Графовое представление белка** ℹ️")
            with g_sub2:
                st.selectbox(
                    "Разметка:", ["ForceAtlas2", "Circular", "Random"],
                    label_visibility="collapsed", index=0, disabled=True, key="layout_opt_panel"
                )
            
            # 1. Инициализируем плеер и ЧИТАЕМ ИЗ ФАЙЛА НА ДИСКЕ (как и левое окно)
            view_graph = py3Dmol.view(width=540, height=400)
            with open(input_path, "r") as f:
                graph_pdb_data = f.read()
            view_graph.addModel(graph_pdb_data, "pdb")
            
            # Жестко вычищаем дефолтные химические палочки и ставим чёрный фон
            view_graph.setStyle({}, {})
            view_graph.setBackgroundColor('black')
            
            total_resi = residues_count
            ca_coords = []
            hex_colors = []
            
            # 2. Парсер Biopython тоже читает этот же файл input_path
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", input_path)
            ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]
            
            # Генерируем палитру спектра
            for idx in range(total_resi):
                if total_resi > 1:
                    hue = 275 - int((idx / (total_resi - 1)) * 275)
                else:
                    hue = 275
                h = hue / 60.0
                x = 1.0 - abs((h % 2.0) - 1.0)
                if 0 <= h < 1: r, g, b = 1.0, x, 0.0
                elif 1 <= h < 2: r, g, b = x, 1.0, 0.0
                elif 2 <= h < 3: r, g, b = 0.0, 1.0, x
                elif 3 <= h < 4: r, g, b = 0.0, x, 1.0
                elif 4 <= h < 5: r, g, b = x, 0.0, 1.0
                else: r, g, b = 1.0, 0.0, x
                
                hex_color = f"0x{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                hex_colors.append(hex_color)
                
                if idx < len(ca_atoms):
                    ca_coords.append(ca_atoms[idx].get_coord().tolist())
                    
                # Отрисовываем цветные сферы узлов графа
                view_graph.setStyle(
                    {"resi": idx + 1, "atom": "CA"},
                    {"sphere": {"radius": 0.6, "color": hex_color}}
                )

            # 3. Натягиваем направленные стрелки-ребра по точным XYZ координатам
            for idx in range(1, len(ca_coords)):
                p1 = ca_coords[idx - 1]
                p2 = ca_coords[idx]
                view_graph.addArrow({
                    "start": {"x": p1[0], "y": p1[1], "z": p1[2]},
                    "end": {"x": p2[0], "y": p2[1], "z": p2[2]},
                    "radius": 0.10,
                    "coneRadius": 0.26,
                    "color": hex_colors[idx - 1],
                    "toColor": hex_colors[idx]
                })

            view_graph.zoomTo()
            components.html(view_graph._make_html(), height=400, width=540)

        
        # Текстовые подписи строго под рамками контейнеров
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-between; font-size: 0.8rem; color: #8b949e; margin-top: 5px;'>
                <div>🔵 Сферы = CA-узлы графа ({residues_count}) <br> ➖ Линии = пептидные ребра</div>
                <div style='text-align: right;'>Цвет ребер плавно переходит между парами смежных узлов</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Метрики качества**")
    
    # Сетка из 6 колонок под карточки дашборда
    met_col1, met_col2, met_col3, met_col4, met_col5, met_col6 = st.columns(6)
    
    # Вспомогательный генератор HTML-карточек
    def generate_metric_card(title, value, subtext, color="#e2e8f0"):
        return f"""
        <div style='background-color: #161b22; padding: 12px; border-radius: 6px; 
        border: 1px solid #30363d; text-align: center; min-height: 110px;'>
            <div style='font-size: 0.75rem; color: #8b949e; margin-bottom: 6px;'>{title}</div>
            <div style='font-size: 1.25rem; font-weight: bold; color: {color}; margin-bottom: 4px;'>{value}</div>
            <div style='font-size: 0.7rem; color: #484f58;'>{subtext}</div>
        </div>
        """

    # Если реконструкция завершена — берем посчитанные метрики, иначе выводим прочерки
    m = dashboard_metrics if is_reconstructed else {
        "rmsd_before": "—", "rmsd_after": "—", "mean_shift": "—", 
        "max_shift": "—", "tm_before": "—", "tm_after": "—"
    }

    with met_col1: st.markdown(generate_metric_card("Начальный шум (RMSD)", m["rmsd_before"], "До реконструкции", "#f87171"), unsafe_allow_html=True)
    with met_col2: st.markdown(generate_metric_card("Финальный RMSD", m["rmsd_after"], "После реконструкции", "#4ade80"), unsafe_allow_html=True)
    with met_col3: st.markdown(generate_metric_card("Mean Atom Shift", m["mean_shift"], "Среднее смещение атомов", "#60a5fa"), unsafe_allow_html=True)
    with met_col4: st.markdown(generate_metric_card("Max Atom Shift", m["max_shift"], "Максимальное смещение", "#fb923c"), unsafe_allow_html=True)
    with met_col5: st.markdown(generate_metric_card("TM-score (до)", m["tm_before"], "До реконструкции"), unsafe_allow_html=True)
    with met_col6: st.markdown(generate_metric_card("TM-score (после)", m["tm_after"], "После реконструкции"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Раскрывающийся аккордеон сравнения структур
    with st.expander("Сравнение структур"):
        if is_reconstructed:
            st.success("📐 Пространственное совмещение (Superposition) выполнено с помощью алгоритма Кабша!")
            cmp_col1, cmp_col2 = st.columns(2)
            
            with cmp_col1:
                st.caption("🟢 Зеленый = Исходный идеальный эталон | 🔴 Красный = Восстановленный ProtEye EGNN")
                # Рендерим вашу кастомную оверлей-сцену наложения
                show_overlay_custom(input_path, output_path)
        else:
            st.warning("Запустите реконструкцию на левой панели, чтобы разблокировать сравнительный 3D-анализ и загрузку файла.")
else:
    st.info("Левая панель полностью настроена. Ожидание загрузки структуры для вывода окон визуализации и метрик.")
