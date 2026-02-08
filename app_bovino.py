import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from bovino_analise import analisar_bovino, gerar_curva_crescimento


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
st.caption(
    "Segmentação aprimorada + análise de características para estimativa de peso e raça "
    "com confiança calibrada até 90%."
)

with st.sidebar:
    st.header("Configuração da análise")
    idade = st.slider("Idade do animal (dias)", min_value=0, max_value=450, value=120)
    st.markdown(
        "**Observação técnica**: estimativa por visão computacional. "
        "Use pesagem real e avaliação zootécnica para validação final."
    )

upload = st.file_uploader("Selecione uma foto do bovino", type=["jpg", "jpeg", "png"])

if upload:
    imagem_bgr = carregar_imagem(upload)
    resultado = analisar_bovino(imagem_bgr, idade)

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

    st.subheader("Visualização da área detectada")
    st.image(cv2.cvtColor(overlay_mascara(imagem_bgr, resultado.mascara), cv2.COLOR_BGR2RGB), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peso estimado", f"{resultado.peso_estimado} kg")
    m2.metric("Raça estimada", resultado.raca_estimada)
    m3.metric("Confiança peso", f"{int(resultado.confianca_peso * 100)}%")
    m4.metric("Confiança raça", f"{int(resultado.confianca_raca * 100)}%")

    st.progress(resultado.confianca_peso, text="Nível de confiança da avaliação de peso")

    st.subheader("Indicadores de qualidade da segmentação")
    st.write(
        {
            "Área relativa do corpo": resultado.area_relativa,
            "Perímetro relativo": resultado.perimetro_relativo,
            "Solidez do contorno": resultado.solidez,
            "Razão largura/altura do bounding box": resultado.razao_bbox,
            "Faixa etária analisada": f"0-450 dias (idade informada: {idade})",
        }
    )

    curva = gerar_curva_crescimento(resultado.peso_estimado, idade)
    st.subheader("Curva de crescimento estimada até 450 dias")
    st.line_chart(curva, x="Idade (dias)", y="Peso estimado (kg)")
else:
    st.info("Faça upload de uma imagem para iniciar a análise.")
