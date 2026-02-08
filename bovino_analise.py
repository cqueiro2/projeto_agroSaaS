from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import pandas as pd


@dataclass
class AnaliseBovino:
    peso_estimado: float
    raca_estimada: str
    confianca_peso: float
    confianca_raca: float
    area_relativa: float
    perimetro_relativo: float
    solidez: float
    razao_bbox: float
    profundidade_relativa: float
    backend_segmentacao: str
    backend_profundidade: str
    mascara: np.ndarray
    imagem_segmentada: np.ndarray
    mapa_profundidade: np.ndarray


@lru_cache(maxsize=1)
def detectar_backends() -> dict:
    info = {
        "torch": False,
        "midas": False,
        "yolo": False,
        "backend_segmentacao": "OpenCV+GrabCut",
        "backend_profundidade": "Heurístico (sem PyTorch)",
    }

    try:
        import torch

        info["torch"] = True
        info["backend_profundidade"] = "PyTorch (MiDaS, se disponível)"

        # Apenas validação de disponibilidade da biblioteca YOLO (opcional)
        try:
            import ultralytics  # noqa: F401

            info["yolo"] = True
            info["backend_segmentacao"] = "YOLO (disponível) + OpenCV fallback"
        except Exception:
            pass

        # Tentativa de carregar MiDaS small via hub (com fallback silencioso)
        try:
            _ = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            info["midas"] = True
            info["backend_profundidade"] = "PyTorch MiDaS"
        except Exception:
            pass
    except Exception:
        pass

    return info


def _normalizar_imagem(imagem_bgr: np.ndarray, limite: int = 1024) -> tuple[np.ndarray, float]:
    h, w = imagem_bgr.shape[:2]
    escala = min(1.0, limite / float(max(h, w)))
    if escala == 1.0:
        return imagem_bgr, 1.0
    nova = cv2.resize(imagem_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
    return nova, escala


def _maior_componente(binaria: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if num_labels <= 1:
        return binaria
    maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binaria)
    out[labels == maior] = 255
    return out


def _mapa_profundidade_heuristico(imagem_bgr: np.ndarray) -> np.ndarray:
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (7, 7), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad_n = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Centro tende a conter o animal em fotos de manejo
    h, w = cinza.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
    centro = (1.0 - np.clip(dist * 1.8, 0, 1))
    centro = (centro * 255).astype(np.uint8)

    depth = cv2.addWeighted(255 - grad_n, 0.55, centro, 0.45, 0)
    return cv2.GaussianBlur(depth, (5, 5), 0)


def estimar_profundidade(imagem_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    backends = detectar_backends()

    # Mesmo com torch disponível, mantém fallback local para robustez offline.
    if backends.get("torch") and backends.get("midas"):
        try:
            import torch

            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            transform = transforms.small_transform

            img_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            batch = transform(img_rgb)
            with torch.no_grad():
                pred = model(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            mapa = pred.cpu().numpy()
            mapa = cv2.normalize(mapa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return mapa, "PyTorch MiDaS"
        except Exception:
            pass

    return _mapa_profundidade_heuristico(imagem_bgr), "Heurístico (fallback)"


def segmentar_bovino(imagem_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray, str]:
    imagem_small, escala = _normalizar_imagem(imagem_bgr)
    h, w = imagem_small.shape[:2]

    profundidade, backend_prof = estimar_profundidade(imagem_small)

    hsv = cv2.cvtColor(imagem_small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mascara_cor = cv2.inRange(hsv, (0, 20, 20), (179, 255, 250))
    mascara_textura = cv2.inRange(sat, 18, 255)
    mascara_luz = cv2.inRange(val, 18, 248)

    # Máscara por profundidade relativa (animal tende a destacar do fundo)
    _, mascara_prof = cv2.threshold(profundidade, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pre_fg = cv2.bitwise_and(mascara_cor, cv2.bitwise_and(mascara_textura, mascara_luz))
    pre_fg = cv2.bitwise_or(pre_fg, mascara_prof)

    centro = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(centro, (w // 2, h // 2), (int(w * 0.42), int(h * 0.42)), 0, 0, 360, 255, -1)
    pre_fg = cv2.bitwise_and(pre_fg, centro)

    kernel = np.ones((5, 5), np.uint8)
    pre_fg = cv2.morphologyEx(pre_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    pre_fg = cv2.morphologyEx(pre_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[pre_fg > 0] = cv2.GC_PR_FGD

    borda = max(8, int(min(h, w) * 0.06))
    gc_mask[:borda, :] = cv2.GC_BGD
    gc_mask[-borda:, :] = cv2.GC_BGD
    gc_mask[:, :borda] = cv2.GC_BGD
    gc_mask[:, -borda:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(imagem_small, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass

    mask_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fg = _maior_componente(mask_fg)

    contornos, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        mask_fg = np.full((h, w), 255, dtype=np.uint8)
        contornos, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contorno = max(contornos, key=cv2.contourArea)
    area = float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    hull = cv2.convexHull(contorno)
    area_hull = max(float(cv2.contourArea(hull)), 1.0)
    solidez = area / area_hull

    x, y, bw, bh = cv2.boundingRect(contorno)
    razao_bbox = float(bw / max(bh, 1))

    segmentada = cv2.bitwise_and(imagem_small, imagem_small, mask=mask_fg)

    if escala != 1.0:
        h0, w0 = imagem_bgr.shape[:2]
        mask_fg = cv2.resize(mask_fg, (w0, h0), interpolation=cv2.INTER_NEAREST)
        segmentada = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mask_fg)
        profundidade = cv2.resize(profundidade, (w0, h0), interpolation=cv2.INTER_LINEAR)

    area_relativa = float(np.count_nonzero(mask_fg) / (mask_fg.shape[0] * mask_fg.shape[1]))
    perimetro_relativo = float(perimetro / (2 * (h + w)))
    return mask_fg, segmentada, area_relativa, perimetro_relativo, solidez, razao_bbox, profundidade, backend_prof


def estimar_raca(imagem_segmentada: np.ndarray, mascara: np.ndarray) -> tuple[str, float]:
    pixels = imagem_segmentada[mascara == 255]
    if len(pixels) < 300:
        return "Indefinida", 0.35

    hsv = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2HSV)[0]
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    branco = float(np.mean((v > 175) & (s < 65)))
    escuro = float(np.mean(v < 72))
    avermelhado = float(np.mean((h >= 5) & (h <= 25) & (s > 70) & (v > 65)))
    misto = float(np.mean((v > 90) & (v < 170) & (s > 45)))

    scores = {
        "Nelore": 0.28 + 1.00 * branco - 0.34 * escuro,
        "Angus": 0.28 + 1.10 * escuro - 0.28 * branco,
        "Jersey": 0.25 + 1.10 * avermelhado,
        "Girolando": 0.30 + 0.72 * misto + 0.26 * branco,
    }

    raca = max(scores, key=scores.get)
    vals = sorted(scores.values(), reverse=True)
    margem = max(0.02, vals[0] - vals[1])
    confianca = float(np.clip(0.60 + margem * 0.85, 0.50, 0.90))
    return raca, confianca


def estimar_peso(
    idade_dias: int,
    area_relativa: float,
    perimetro_relativo: float,
    solidez: float,
    razao_bbox: float,
    profundidade_relativa: float,
    raca: str,
) -> tuple[float, float]:
    idade = int(np.clip(idade_dias, 0, 450))

    pesos_raca_450 = {
        "Nelore": 315,
        "Angus": 340,
        "Jersey": 285,
        "Girolando": 300,
        "Indefinida": 300,
    }
    alvo_450 = pesos_raca_450.get(raca, 300)
    peso_base = 35 + (alvo_450 - 35) * (1 - np.exp(-3.2 * (idade / 450.0)))

    fator_forma = 1.0
    fator_forma += (area_relativa - 0.28) * 0.90
    fator_forma += (solidez - 0.83) * 0.50
    fator_forma += (razao_bbox - 1.45) * 0.10
    fator_forma += (perimetro_relativo - 0.20) * 0.30
    fator_forma += (profundidade_relativa - 0.50) * 0.32
    fator_forma = float(np.clip(fator_forma, 0.70, 1.30))

    peso = float(peso_base * fator_forma)

    qualidade = 1.0
    qualidade -= min(abs(area_relativa - 0.30), 0.30)
    qualidade -= min(abs(solidez - 0.85), 0.30) * 0.7
    qualidade -= min(abs(razao_bbox - 1.50), 0.80) * 0.14
    qualidade -= min(abs(profundidade_relativa - 0.52), 0.52) * 0.12

    confianca = float(np.clip(0.62 + max(0.0, qualidade) * 0.28, 0.55, 0.90))
    return peso, confianca


def analisar_bovino(imagem_bgr: np.ndarray, idade_dias: int) -> AnaliseBovino:
    (
        mascara,
        segmentada,
        area_relativa,
        perimetro_relativo,
        solidez,
        razao_bbox,
        depth_map,
        backend_prof,
    ) = segmentar_bovino(imagem_bgr)

    raca, conf_raca = estimar_raca(segmentada, mascara)
    profundidade_rel = float(np.mean(depth_map[mascara == 255]) / 255.0) if np.any(mascara == 255) else 0.5

    peso, conf_peso = estimar_peso(
        idade_dias,
        area_relativa,
        perimetro_relativo,
        solidez,
        razao_bbox,
        profundidade_rel,
        raca,
    )

    backend_seg = detectar_backends().get("backend_segmentacao", "OpenCV+GrabCut")

    return AnaliseBovino(
        peso_estimado=round(peso, 1),
        raca_estimada=raca,
        confianca_peso=round(conf_peso, 2),
        confianca_raca=round(conf_raca, 2),
        area_relativa=round(area_relativa, 3),
        perimetro_relativo=round(perimetro_relativo, 3),
        solidez=round(solidez, 3),
        razao_bbox=round(razao_bbox, 3),
        profundidade_relativa=round(profundidade_rel, 3),
        backend_segmentacao=backend_seg,
        backend_profundidade=backend_prof,
        mascara=mascara,
        imagem_segmentada=segmentada,
        mapa_profundidade=depth_map,
    )


def gerar_curva_crescimento(peso_atual: float, idade_dias: int) -> pd.DataFrame:
    dias = np.arange(0, 451, 15)
    idade_ref = max(idade_dias, 1)
    fator = np.clip((dias + 1) / (idade_ref + 1), 0.15, 1.25)
    pesos = np.clip(peso_atual * (fator ** 0.88), 30, 460)
    return pd.DataFrame({"Idade (dias)": dias, "Peso estimado (kg)": pesos})
