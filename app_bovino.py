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


st.set_page_config(page_title="AgroSaaS Bovino", page_icon="🐄", layout="wide")


def carregar_imagem(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def overlay_mascara(imagem_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    overlay = imagem_bgr.copy()
    overlay[mascara == 255] = cv2.addWeighted(
        imagem_bgr[mascara == 255], 0.55, np.full_like(imagem_bgr[mascara == 255], (20, 160, 20)), 0.45, 0
    )
    return overlay


st.title("🐄 AgroSaaS - Avaliação Bovina por Imagem")
st.caption("Análise automática de raça, idade, peso e projeção diária de ganho para bovinos.")

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
    st.dataframe(proj, use_container_width=True)

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
