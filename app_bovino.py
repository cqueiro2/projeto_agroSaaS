import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from bovino_analise import (
    analisar_bovino,
    detectar_backends,
    gerar_curva_crescimento,
    gerar_projecao_peso_diaria,
)


st.set_page_config(page_title="Bovino Vision AI", page_icon="🐄", layout="wide")


def carregar_imagem(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def overlay_mascara(imagem_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    overlay = imagem_bgr.copy()
    overlay[mascara == 255] = cv2.addWeighted(
        imagem_bgr[mascara == 255], 0.55, np.full_like(imagem_bgr[mascara == 255], (20, 160, 20)), 0.45, 0
    )
    return overlay


def aplicar_estilo() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6f8fb 0%, #eef2f5 100%);
            color: #13202b;
        }
        .main .block-container {
            padding-top: 1.0rem;
            padding-bottom: 7rem;
            max-width: 1100px;
        }
        .bv-header {
            background: white;
            border-radius: 20px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 6px 20px rgba(18, 33, 53, 0.08);
            margin-bottom: 12px;
        }
        .bv-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .bv-logo {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: #1f6b2d;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 26px;
        }
        .bv-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: #1e5f2a;
            line-height: 1.1;
        }
        .bv-subtitle {
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            color: #5d6f85;
            font-weight: 600;
        }
        .bv-user {
            width: 44px;
            height: 44px;
            border-radius: 999px;
            background: #eef1f5;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            color: #506074;
        }
        .bv-hero {
            position: relative;
            border-radius: 22px;
            min-height: 350px;
            overflow: hidden;
            background: linear-gradient(120deg, rgba(29, 81, 34, 0.90), rgba(48, 78, 25, 0.74));
            margin-bottom: 14px;
            padding: 1.2rem;
        }
        .bv-badge {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            background: rgba(35, 122, 53, 0.86);
            border: 2px solid rgba(173, 234, 183, 0.38);
            color: #f0fff0;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: 0.03em;
        }
        .bv-hero-note {
            margin-top: 8px;
            color: #e4f2e5;
            font-size: 0.95rem;
            max-width: 480px;
        }
        .bv-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 5px 18px rgba(12, 25, 41, 0.08);
            margin-bottom: 10px;
        }
        .bv-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            background: #f4f7fa;
            border-top: 1px solid #dce3eb;
            padding: 0.65rem;
            z-index: 999;
        }
        .bv-nav {
            max-width: 780px;
            margin: auto;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            text-align: center;
            color: #7f8ea3;
            font-weight: 600;
        }
        .bv-nav .active { color: #1f6b2d; }
        @media (max-width: 760px) {
            .bv-title { font-size: 1.28rem; }
            .bv-subtitle { font-size: 0.78rem; }
            .bv-hero { min-height: 300px; }
            .main .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header_principal() -> None:
    st.markdown(
        """
        <div class="bv-header">
            <div class="bv-brand">
                <div class="bv-logo">📈</div>
                <div>
                    <div class="bv-title">Bovino Vision AI</div>
                    <div class="bv-subtitle">MONITORAMENTO INTELIGENTE</div>
                </div>
            </div>
            <div class="bv-user">👤</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bloco_hero() -> None:
    st.markdown(
        """
        <div class="bv-hero">
            <div class="bv-badge">● SISTEMA PRONTO PARA ANÁLISE</div>
            <div class="bv-hero-note">Carregue uma imagem de bovino para segmentação automática e leitura de profundidade com projeção de peso.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rodape_navegacao() -> None:
    st.markdown(
        """
        <div class="bv-footer">
            <div class="bv-nav">
                <div class="active">🏠 Início</div>
                <div>🕘 Histórico</div>
                <div>📊 Análise</div>
                <div>⚙️ Ajustes</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


aplicar_estilo()
header_principal()
bloco_hero()

st.markdown('<div class="bv-card">', unsafe_allow_html=True)
with st.sidebar:
    st.header("Configuração da análise")
    usar_idade_manual = st.checkbox("Informar idade manualmente", value=False)
    idade_manual = st.slider("Idade manual (dias)", min_value=0, max_value=450, value=120, disabled=not usar_idade_manual)

    info = detectar_backends()
    st.markdown("### Backends detectados")
    st.write(
        {
            "Segmentação": info["backend_segmentacao"],
            "Profundidade": info["backend_profundidade"],
            "PyTorch disponível": info["torch"],
            "YOLO disponível": info["yolo"],
        }
    )
    st.markdown("**Execução recomendada**: `streamlit run app_bovino.py`.")

upload = st.file_uploader("Selecione uma foto do bovino", type=["jpg", "jpeg", "png"])

if upload:
    imagem_bgr = carregar_imagem(upload)
    idade_entrada = idade_manual if usar_idade_manual else None
    resultado = analisar_bovino(imagem_bgr, idade_entrada)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Imagem original")
        st.image(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Máscara de segmentação")
        st.image(resultado.mascara, clamp=True, use_container_width=True)

    with col3:
        st.subheader("Animal isolado")
        st.image(cv2.cvtColor(resultado.imagem_segmentada, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader(f"Mapa de profundidade ({resultado.backend_profundidade})")
    st.image(resultado.mapa_profundidade, clamp=True, use_container_width=True)

    st.subheader("Resultado automático (formato solicitado)")
    st.markdown(
        f"""
- **Raça estimada:** {resultado.raca_estimada}
- **Idade estimada (dias):** {resultado.idade_estimada_dias}
- **Peso estimado (kg) / convertido para arrobas:** {resultado.peso_estimado} kg / {resultado.peso_arroba} @
- **Faixa de variação (kg):** {resultado.faixa_min_kg} a {resultado.faixa_max_kg}
- **Projeção de peso por dia:** ganho médio de **{resultado.ganho_dia_kg} kg/dia** (próximos 30 dias)
- **Nível de confiança (%):** Raça {int(resultado.confianca_raca*100)}% | Idade {int(resultado.confianca_idade*100)}% | Peso {int(resultado.confianca_peso*100)}%
- **Observações adicionais relevantes:** Segmentação por cor/profundidade com backend {resultado.backend_segmentacao}; profundidade via {resultado.backend_profundidade}.
"""
    )

    proj = gerar_projecao_peso_diaria(resultado.peso_estimado, resultado.idade_estimada_dias, horizonte_dias=30)
    st.subheader("Projeção de peso por dia (próximos 30 dias)")
    st.line_chart(proj, x="Dia futuro", y=["Peso estimado (kg)", "Peso mínimo (kg)", "Peso máximo (kg)"])

    st.subheader("Curva de crescimento estimada até 450 dias")
    curva = gerar_curva_crescimento(resultado.peso_estimado, resultado.idade_estimada_dias)
    st.line_chart(curva, x="Idade (dias)", y="Peso estimado (kg)")

    st.subheader("Indicadores técnicos")
    st.write(
        {
            "Área relativa": resultado.area_relativa,
            "Perímetro relativo": resultado.perimetro_relativo,
            "Solidez": resultado.solidez,
            "Razão largura/altura": resultado.razao_bbox,
            "Profundidade relativa": resultado.profundidade_relativa,
        }
    )

    st.subheader("Overlay da área detectada")
    st.image(cv2.cvtColor(overlay_mascara(imagem_bgr, resultado.mascara), cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Faça upload de uma imagem para iniciar a análise.")

st.markdown("</div>", unsafe_allow_html=True)
rodape_navegacao()
