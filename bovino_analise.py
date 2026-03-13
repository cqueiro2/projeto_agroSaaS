from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pandas as pd


COCO_COW_CLASS_ID = 19


@dataclass
class AnaliseBovino:
    raca_estimada: str
    idade_estimada_dias: int
    peso_estimado: float
    peso_arroba: float
    faixa_min_kg: float
    faixa_max_kg: float
    ganho_dia_kg: float
    confianca_raca: float
    confianca_idade: float
    confianca_peso: float
    backend_segmentacao: str
    backend_profundidade: str
    mascara: np.ndarray
    imagem_segmentada: np.ndarray
    imagem_final_det: np.ndarray
    deteccoes_finais: list[dict[str, Any]]


@lru_cache(maxsize=1)
def detectar_backends() -> dict:
    info = {
        "torch": False,
        "midas": False,
        "yolo": False,
        "backend_segmentacao": "Heurístico OpenCV",
        "backend_profundidade": "Heurístico",
    }
    try:
        import torch  # noqa: F401

        info["torch"] = True
        try:
            import ultralytics  # noqa: F401

            info["yolo"] = True
            info["backend_segmentacao"] = "YOLO (classe bovino)"
        except Exception:
            pass
    except Exception:
        pass
    return info


@lru_cache(maxsize=1)
def _load_yolo_model():
    try:
        from ultralytics import YOLO

        return YOLO("yolov8n.pt")
    except Exception:
        return None


def _segmentacao_fallback(imagem_bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = imagem_bgr.shape[:2]
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = cv2.inRange(hsv, (0, 20, 20), (179, 255, 250))
    mask = cv2.bitwise_and(mask, cv2.inRange(s, 15, 255))
    mask = cv2.bitwise_and(mask, cv2.inRange(v, 15, 245))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    deteccoes: list[dict[str, Any]] = []
    out_mask = np.zeros((h, w), dtype=np.uint8)

    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.drawContours(out_mask, [cnt], -1, 255, cv2.FILLED)
        conf = float(np.clip(cv2.contourArea(cnt) / max(h * w, 1), 0.35, 0.89))
        deteccoes.append(
            {
                "classe": "bovino",
                "confianca": conf,
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            }
        )

    return out_mask, deteccoes


def _render_final(imagem_bgr: np.ndarray, deteccoes: list[dict[str, Any]], mascara: np.ndarray | None = None) -> np.ndarray:
    out = imagem_bgr.copy()
    for d in deteccoes:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["confianca"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (38, 176, 74), 2)
        label = f"bovino {conf:.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (38, 176, 74), 2)
    if mascara is not None and np.any(mascara == 255):
        edges = cv2.Canny(mascara, 80, 160)
        out[edges > 0] = (36, 220, 116)
    return out


def detectar_bovinos_final(imagem_bgr: np.ndarray, conf_min: float = 0.25) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str]:
    model = _load_yolo_model()
    if model is not None:
        try:
            preds = model.predict(imagem_bgr, conf=conf_min, verbose=False)
            p = preds[0]
            h, w = imagem_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            deteccoes: list[dict[str, Any]] = []

            if p.boxes is not None and len(p.boxes) > 0:
                cls_arr = p.boxes.cls.cpu().numpy().astype(int)
                conf_arr = p.boxes.conf.cpu().numpy().astype(float)
                xyxy_arr = p.boxes.xyxy.cpu().numpy().astype(int)

                for i, cid in enumerate(cls_arr):
                    if cid != COCO_COW_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = xyxy_arr[i].tolist()
                    conf = float(conf_arr[i])
                    deteccoes.append(
                        {
                            "classe": "bovino",
                            "confianca": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # Se houver máscaras prontas do modelo, usa somente as da classe bovino
            if getattr(p, "masks", None) is not None and p.masks is not None and len(deteccoes) > 0:
                try:
                    m = p.masks.data.cpu().numpy()
                    cls_arr = p.boxes.cls.cpu().numpy().astype(int)
                    for i, cid in enumerate(cls_arr):
                        if cid == COCO_COW_CLASS_ID:
                            mi = cv2.resize((m[i] > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                            mask = cv2.bitwise_or(mask, mi)
                except Exception:
                    pass

            # limpeza final apenas
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            final = _render_final(imagem_bgr, deteccoes, mask)
            return final, mask, deteccoes, "YOLO"
        except Exception:
            pass

    mask, deteccoes = _segmentacao_fallback(imagem_bgr)
    final = _render_final(imagem_bgr, deteccoes, mask)
    return final, mask, deteccoes, "Heurístico"


def processar_frame_video_final(frame_bgr: np.ndarray, conf_min: float = 0.25) -> np.ndarray:
    """Processa um frame de vídeo e retorna APENAS a renderização final limpa."""
    final, _, _, _ = detectar_bovinos_final(frame_bgr, conf_min=conf_min)
    return final


def estimar_raca_por_cor(imagem_segmentada: np.ndarray, mascara: np.ndarray) -> tuple[str, float]:
    pixels = imagem_segmentada[mascara == 255]
    if len(pixels) < 50:
        return "Indefinida", 0.40
    hsv = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2HSV)[0]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    branco = float(np.mean((v > 175) & (s < 65)))
    escuro = float(np.mean(v < 72))
    avermelhado = float(np.mean((h >= 5) & (h <= 25) & (s > 70) & (v > 65)))
    scores = {
        "Nelore": 0.30 + 1.00 * branco - 0.30 * escuro,
        "Angus": 0.30 + 1.05 * escuro - 0.20 * branco,
        "Jersey": 0.25 + 1.10 * avermelhado,
        "Girolando": 0.32 + 0.25 * branco + 0.20 * (1 - escuro),
    }
    raca = max(scores, key=scores.get)
    vals = sorted(scores.values(), reverse=True)
    conf = float(np.clip(0.58 + (vals[0] - vals[1]) * 0.9, 0.50, 0.90))
    return raca, conf


def estimar_idade_dias(area_relativa: float, conf_det: float) -> tuple[int, float]:
    maturidade = np.clip((area_relativa - 0.05) / 0.55, 0, 1)
    idade = int(round(maturidade * 450))
    conf = float(np.clip(0.55 + 0.25 * conf_det + 0.15 * maturidade, 0.50, 0.90))
    return idade, conf


def estimar_peso(idade_dias: int, area_relativa: float, conf_det: float, raca: str) -> tuple[float, float]:
    pesos_raca_450 = {"Nelore": 315, "Angus": 340, "Jersey": 285, "Girolando": 300, "Indefinida": 300}
    alvo = pesos_raca_450.get(raca, 300)
    idade = int(np.clip(idade_dias, 0, 450))
    base = 35 + (alvo - 35) * (1 - np.exp(-3.1 * (idade / 450.0)))
    ajuste = float(np.clip(0.78 + area_relativa * 1.15, 0.70, 1.30))
    peso = base * ajuste
    conf = float(np.clip(0.58 + 0.24 * conf_det + 0.08 * (1 - abs(area_relativa - 0.30)), 0.55, 0.90))
    return float(peso), conf


def gerar_projecao_peso_diaria(peso_atual: float, idade_dias: int, horizonte_dias: int = 30) -> pd.DataFrame:
    dias = np.arange(0, horizonte_dias + 1)
    idade = np.clip(idade_dias + dias, 0, 450)
    maturidade = idade / 450.0
    ganho = np.clip(0.95 * (1 - maturidade) + 0.15, 0.15, 1.1)
    pesos = peso_atual + np.cumsum(ganho) - ganho[0]
    var = np.clip(0.08 - maturidade * 0.03, 0.04, 0.08)
    return pd.DataFrame(
        {
            "Dia futuro": dias,
            "Peso estimado (kg)": pesos,
            "Peso mínimo (kg)": pesos * (1 - var),
            "Peso máximo (kg)": pesos * (1 + var),
        }
    )


def gerar_curva_crescimento(peso_atual: float, idade_dias: int) -> pd.DataFrame:
    dias = np.arange(0, 451, 15)
    fator = np.clip((dias + 1) / max(idade_dias + 1, 1), 0.15, 1.25)
    pesos = np.clip(peso_atual * (fator**0.88), 30, 460)
    return pd.DataFrame({"Idade (dias)": dias, "Peso estimado (kg)": pesos})


def analisar_bovino(imagem_bgr: np.ndarray, idade_dias: int | None = None) -> AnaliseBovino:
    final_det, mask, deteccoes, backend_seg = detectar_bovinos_final(imagem_bgr)
    seg = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mask)
    conf_det = float(max([d["confianca"] for d in deteccoes], default=0.45))
    area_rel = float(np.count_nonzero(mask) / max(mask.size, 1))

    raca, conf_raca = estimar_raca_por_cor(seg, mask)
    idade_auto, conf_idade = estimar_idade_dias(area_rel, conf_det)
    idade_final = idade_auto if idade_dias is None else int(np.clip(idade_dias, 0, 450))
    peso, conf_peso = estimar_peso(idade_final, area_rel, conf_det, raca)

    proj = gerar_projecao_peso_diaria(peso, idade_final, horizonte_dias=30)
    ganho = float(proj["Peso estimado (kg)"].iloc[1] - proj["Peso estimado (kg)"].iloc[0]) if len(proj) > 1 else 0.0

    return AnaliseBovino(
        raca_estimada=raca,
        idade_estimada_dias=int(idade_final),
        peso_estimado=round(float(peso), 1),
        peso_arroba=round(float(peso) / 15.0, 2),
        faixa_min_kg=round(float(proj["Peso mínimo (kg)"].iloc[0]), 1),
        faixa_max_kg=round(float(proj["Peso máximo (kg)"].iloc[0]), 1),
        ganho_dia_kg=round(ganho, 3),
        confianca_raca=round(float(conf_raca), 2),
        confianca_idade=round(float(conf_idade), 2),
        confianca_peso=round(float(conf_peso), 2),
        backend_segmentacao=backend_seg,
        backend_profundidade="Oculto (pipeline interno)",
        mascara=mask,
        imagem_segmentada=seg,
        imagem_final_det=final_det,
        deteccoes_finais=deteccoes,
    )
