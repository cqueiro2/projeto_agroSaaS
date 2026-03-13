from __future__ import annotations

import base64
import io

from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)


def _engine():
    from bovino_analise import analisar_bovino, detectar_backends, gerar_curva_crescimento, gerar_projecao_peso_diaria

    return analisar_bovino, detectar_backends, gerar_curva_crescimento, gerar_projecao_peso_diaria


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    import cv2

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_data_uri(image_bgr: np.ndarray) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def gray_to_data_uri(gray: np.ndarray) -> str:
    import cv2

    ok, buffer = cv2.imencode(".png", gray)
    if not ok:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def overlay_mascara(imagem_bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    overlay = imagem_bgr.copy()
    overlay[mascara == 255] = cv2.addWeighted(
        imagem_bgr[mascara == 255],
        0.55,
        np.full_like(imagem_bgr[mascara == 255], (20, 160, 20)),
        0.45,
        0,
    )
    return overlay


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        _, detectar_backends, _, _ = _engine()
        info = detectar_backends()
    except Exception:
        info = {
            "backend_segmentacao": "Indisponível no ambiente",
            "backend_profundidade": "Indisponível no ambiente",
            "torch": False,
            "yolo": False,
        }

    context = {
        "result": None,
        "images": {},
        "projection": [],
        "curve": [],
        "info": info,
        "idade_manual": 120,
        "usar_idade_manual": False,
        "erro": None,
    }

    if request.method == "POST":
        file = request.files.get("imagem")
        usar_idade_manual = request.form.get("usar_idade_manual") == "on"
        idade_manual = int(request.form.get("idade_manual", "120") or 120)

        context["usar_idade_manual"] = usar_idade_manual
        context["idade_manual"] = idade_manual

        try:
            analisar_bovino, _, gerar_curva_crescimento, gerar_projecao_peso_diaria = _engine()
            if file and file.filename:
                img_bytes = file.read()
                imagem_bgr = bytes_to_bgr(img_bytes)
                idade_entrada = idade_manual if usar_idade_manual else None
                analise = analisar_bovino(imagem_bgr, idade_entrada)

                proj = gerar_projecao_peso_diaria(analise.peso_estimado, analise.idade_estimada_dias, horizonte_dias=30)
                curva = gerar_curva_crescimento(analise.peso_estimado, analise.idade_estimada_dias)

                context["result"] = analise
                context["images"] = {
                    "original": bgr_to_data_uri(imagem_bgr),
                    "mask": gray_to_data_uri(analise.mascara),
                    "segmentada": bgr_to_data_uri(analise.imagem_segmentada),
                    "profundidade": gray_to_data_uri(analise.mapa_profundidade),
                    "overlay": bgr_to_data_uri(overlay_mascara(imagem_bgr, analise.mascara)),
                }
                context["projection"] = proj.round(2).to_dict(orient="records")
                context["curve"] = curva.round(2).to_dict(orient="records")
        except Exception as exc:
            context["erro"] = f"Não foi possível processar no ambiente atual: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
