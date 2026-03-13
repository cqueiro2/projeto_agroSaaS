import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from bovino_analise import analisar_bovino, detectar_backends, gerar_projecao_peso_diaria

st.set_page_config(page_title="Bovino Vision AI", page_icon="🐄", layout="wide")


def carregar_imagem(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


st.title("🐄 Bovino Vision AI")
st.caption("Saída limpa: apenas resultado final de detecção bovino (sem mostrar etapas internas).")

with st.sidebar:
    usar_idade_manual = st.checkbox("Informar idade manualmente", value=False)
    idade_manual = st.slider("Idade manual (dias)", 0, 450, 120, disabled=not usar_idade_manual)
    info = detectar_backends()
    st.write({"Segmentação final": info["backend_segmentacao"], "YOLO": info["yolo"]})

upload = st.file_uploader("Selecione imagem", type=["jpg", "jpeg", "png"])
if upload:
    img = carregar_imagem(upload)
    res = analisar_bovino(img, idade_manual if usar_idade_manual else None)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagem original", use_container_width=True)
    c2.image(cv2.cvtColor(res.imagem_final_det, cv2.COLOR_BGR2RGB), caption="Detecções finais bovino", use_container_width=True)

    st.image(res.mascara, caption="Máscara final refinada (opcional)", clamp=True, use_container_width=True)
    st.markdown(
        f"""
- **Raça estimada:** {res.raca_estimada}
- **Idade estimada (dias):** {res.idade_estimada_dias}
- **Peso estimado:** {res.peso_estimado} kg / {res.peso_arroba} arrobas
- **Faixa de variação:** {res.faixa_min_kg} - {res.faixa_max_kg} kg
- **Confiança:** Raça {int(res.confianca_raca*100)}% | Idade {int(res.confianca_idade*100)}% | Peso {int(res.confianca_peso*100)}%
"""
    )
    st.subheader("Detecções finais")
    st.dataframe(res.deteccoes_finais, use_container_width=True)

    proj = gerar_projecao_peso_diaria(res.peso_estimado, res.idade_estimada_dias, 30)
    st.line_chart(proj, x="Dia futuro", y=["Peso estimado (kg)", "Peso mínimo (kg)", "Peso máximo (kg)"])
else:
    st.info("Envie uma imagem para processar.")
