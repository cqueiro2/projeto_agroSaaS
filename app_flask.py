from __future__ import annotations

import base64
import io
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image

from sqlite_store import (
    create_analysis,
    delete_analysis,
    import_dataset_if_empty,
    init_db,
    list_dataset_bovinos,
    read_analyses,
    read_analysis,
    update_analysis,
)

app = Flask(__name__)

init_db()
import_dataset_if_empty()

ALLOWED_MIME = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _engine():
    from bovino_analise import analisar_bovino, detectar_backends, gerar_projecao_peso_diaria

    return analisar_bovino, detectar_backends, gerar_projecao_peso_diaria


def _error_context(msg: str) -> dict[str, Any]:
    return {
        "result": None,
        "images": {},
        "projection": [],
        "info": {"backend_segmentacao": "-", "yolo": False},
        "idade_manual": 120,
        "usar_idade_manual": False,
        "erro": msg,
        "recentes": read_analyses(20),
        "dataset": list_dataset_bovinos(100),
        "registro": None,
    }


def _validate_upload(file) -> tuple[bool, str]:
    if not file or not file.filename:
        return False, "Envie uma imagem JPG/PNG."
    if file.mimetype not in ALLOWED_MIME:
        return False, "Formato inválido. Use JPG ou PNG."
    file.stream.seek(0, io.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return False, "Arquivo muito grande (máximo 8MB)."
    return True, ""


def bytes_to_bgr(file_bytes: bytes):
    import cv2
    import numpy as np

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_data_uri(image_bgr) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"


def gray_to_data_uri(gray) -> str:
    import cv2

    ok, buffer = cv2.imencode(".png", gray)
    if not ok:
        return ""
    return f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"


@app.get("/")
def home():
    return redirect(url_for("read_route"))


@app.route("/create", methods=["GET", "POST"])
def create_route():
    try:
        _, detectar_backends, _ = _engine()
        info = detectar_backends()
    except Exception:
        info = {"backend_segmentacao": "Indisponível no ambiente", "yolo": False}

    context = {
        "result": None,
        "images": {},
        "projection": [],
        "info": info,
        "idade_manual": 120,
        "usar_idade_manual": False,
        "erro": None,
        "recentes": read_analyses(20),
        "dataset": list_dataset_bovinos(100),
        "registro": None,
    }

    if request.method == "POST":
        file = request.files.get("imagem")
        ok, msg = _validate_upload(file)
        if not ok:
            context["erro"] = msg
            return render_template("index.html", **context)

        usar_idade_manual = request.form.get("usar_idade_manual") == "on"
        idade_manual = int(request.form.get("idade_manual", "120") or 120)
        context["usar_idade_manual"] = usar_idade_manual
        context["idade_manual"] = idade_manual

        try:
            analisar_bovino, _, gerar_projecao_peso_diaria = _engine()
            imagem_bgr = bytes_to_bgr(file.read())
            if imagem_bgr is None:
                context["erro"] = "Arquivo inválido. Envie uma imagem JPG/PNG válida."
                return render_template("index.html", **context)

            idade_entrada = idade_manual if usar_idade_manual else None
            analise = analisar_bovino(imagem_bgr, idade_entrada)
            proj = gerar_projecao_peso_diaria(analise.peso_estimado, analise.idade_estimada_dias, horizonte_dias=30)

            images = {
                "original": bgr_to_data_uri(imagem_bgr),
                "final": bgr_to_data_uri(analise.imagem_final_det),
                "mask": gray_to_data_uri(analise.mascara),
            }

            analysis_id = create_analysis(analise, proj.round(2).to_dict(orient="records"), images)

            context["result"] = analise
            context["images"] = images
            context["projection"] = proj.round(2).to_dict(orient="records")
            context["recentes"] = read_analyses(20)
            context["registro"] = read_analysis(analysis_id)
        except Exception as exc:
            context["erro"] = f"Não foi possível processar no ambiente atual: {exc}"

    return render_template("index.html", **context)


@app.get("/read")
def read_route():
    try:
        _, detectar_backends, _ = _engine()
        info = detectar_backends()
    except Exception:
        info = {"backend_segmentacao": "Indisponível no ambiente", "yolo": False}

    return render_template(
        "index.html",
        result=None,
        images={},
        projection=[],
        info=info,
        idade_manual=120,
        usar_idade_manual=False,
        erro=None,
        recentes=read_analyses(20),
        dataset=list_dataset_bovinos(100),
        registro=None,
    )


@app.route("/read/<int:analysis_id>", methods=["GET"])
def read_one_route(analysis_id: int):
    record = read_analysis(analysis_id)
    if not record:
        return jsonify({"ok": False, "error": "Registro não encontrado"}), 404
    return jsonify({"ok": True, "data": record})


@app.route("/update/<int:analysis_id>", methods=["POST"])
def update_route(analysis_id: int):
    idade_real_raw = request.form.get("idade_real", "").strip()
    raca_corrigida = request.form.get("raca_corrigida", "").strip() or None
    peso_medido_raw = request.form.get("peso_medido", "").strip()

    idade_real = int(idade_real_raw) if idade_real_raw else None
    peso_medido = float(peso_medido_raw) if peso_medido_raw else None

    ok = update_analysis(analysis_id, idade_real, raca_corrigida, peso_medido)
    if not ok:
        return jsonify({"ok": False, "error": "Registro não encontrado"}), 404
    return redirect(url_for("read_route"))


@app.route("/delete/<int:analysis_id>", methods=["POST"])
def delete_route(analysis_id: int):
    ok = delete_analysis(analysis_id)
    if not ok:
        return jsonify({"ok": False, "error": "Registro não encontrado"}), 404
    return redirect(url_for("read_route"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
