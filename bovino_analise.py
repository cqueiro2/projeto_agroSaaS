from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd


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
        try:
            import ultralytics  # noqa: F401

            info["yolo"] = True
            info["backend_segmentacao"] = "YOLO (disponível) + OpenCV fallback"
        except Exception:
            pass
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
    return cv2.resize(imagem_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA), escala


def _maior_componente(binaria: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:
        return binaria
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binaria)
    out[labels == i] = 255
    return out


def _mapa_profundidade_heuristico(imagem_bgr: np.ndarray) -> np.ndarray:
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (7, 7), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad_n = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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
            return cv2.normalize(mapa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), "PyTorch MiDaS"
        except Exception:
            pass
    return _mapa_profundidade_heuristico(imagem_bgr), "Heurístico (fallback)"


def segmentar_bovino(imagem_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray, str]:
    imagem_small, escala = _normalizar_imagem(imagem_bgr)
    h, w = imagem_small.shape[:2]
    profundidade, backend_prof = estimar_profundidade(imagem_small)

    hsv = cv2.cvtColor(imagem_small, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mascara_cor = cv2.inRange(hsv, (0, 20, 20), (179, 255, 250))
    mascara_textura = cv2.inRange(sat, 18, 255)
    mascara_luz = cv2.inRange(val, 18, 248)
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
    gc_mask[:borda, :], gc_mask[-borda:, :], gc_mask[:, :borda], gc_mask[:, -borda:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD

    try:
        cv2.grabCut(imagem_small, gc_mask, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
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

    perimetro = float(cv2.arcLength(contorno, True))
    hull = cv2.convexHull(contorno)
    solidez = float(cv2.contourArea(contorno) / max(float(cv2.contourArea(hull)), 1.0))
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
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
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
    conf = float(np.clip(0.60 + max(0.02, vals[0] - vals[1]) * 0.85, 0.50, 0.90))
    return raca, conf


def estimar_idade_dias(area_relativa: float, solidez: float, razao_bbox: float, profundidade_relativa: float) -> tuple[int, float]:
    maturidade = 0.0
    maturidade += np.clip((area_relativa - 0.18) / 0.22, 0, 1) * 0.40
    maturidade += np.clip((solidez - 0.68) / 0.30, 0, 1) * 0.20
    maturidade += np.clip((razao_bbox - 1.05) / 0.85, 0, 1) * 0.20
    maturidade += np.clip((profundidade_relativa - 0.35) / 0.40, 0, 1) * 0.20
    idade = int(np.clip(round(maturidade * 450), 0, 450))
    conf = float(np.clip(0.58 + maturidade * 0.32, 0.50, 0.90))
    return idade, conf


def estimar_peso(idade_dias: int, area_relativa: float, perimetro_relativo: float, solidez: float, razao_bbox: float, profundidade_relativa: float, raca: str) -> tuple[float, float]:
    idade = int(np.clip(idade_dias, 0, 450))
    pesos_raca_450 = {"Nelore": 315, "Angus": 340, "Jersey": 285, "Girolando": 300, "Indefinida": 300}
    alvo_450 = pesos_raca_450.get(raca, 300)
    peso_base = 35 + (alvo_450 - 35) * (1 - np.exp(-3.2 * (idade / 450.0)))

    fator = 1.0 + (area_relativa - 0.28) * 0.90 + (solidez - 0.83) * 0.50 + (razao_bbox - 1.45) * 0.10
    fator += (perimetro_relativo - 0.20) * 0.30 + (profundidade_relativa - 0.50) * 0.32
    fator = float(np.clip(fator, 0.70, 1.30))
    peso = float(peso_base * fator)

    qualidade = 1.0 - min(abs(area_relativa - 0.30), 0.30) - min(abs(solidez - 0.85), 0.30) * 0.7
    qualidade -= min(abs(razao_bbox - 1.50), 0.80) * 0.14 + min(abs(profundidade_relativa - 0.52), 0.52) * 0.12
    conf = float(np.clip(0.62 + max(0.0, qualidade) * 0.28, 0.55, 0.90))
    return peso, conf


def gerar_projecao_peso_diaria(peso_atual: float, idade_dias: int, horizonte_dias: int = 30) -> pd.DataFrame:
    dias = np.arange(0, horizonte_dias + 1)
    idade = np.clip(idade_dias + dias, 0, 450)
    maturidade = idade / 450.0
    ganho_dia = np.clip(0.95 * (1 - maturidade) + 0.15, 0.15, 1.1)
    pesos = peso_atual + np.cumsum(ganho_dia) - ganho_dia[0]
    var_pct = np.clip(0.08 - maturidade * 0.03, 0.04, 0.08)
    peso_min = pesos * (1 - var_pct)
    peso_max = pesos * (1 + var_pct)
    return pd.DataFrame({
        "Dia futuro": dias,
        "Peso estimado (kg)": pesos,
        "Peso mínimo (kg)": peso_min,
        "Peso máximo (kg)": peso_max,
    })


def analisar_bovino(imagem_bgr: np.ndarray, idade_dias: int | None = None) -> AnaliseBovino:
    mascara, segmentada, area_relativa, perimetro_relativo, solidez, razao_bbox, depth_map, backend_prof = segmentar_bovino(imagem_bgr)
    raca, conf_raca = estimar_raca(segmentada, mascara)
    profundidade_rel = float(np.mean(depth_map[mascara == 255]) / 255.0) if np.any(mascara == 255) else 0.5

    idade_auto, conf_idade = estimar_idade_dias(area_relativa, solidez, razao_bbox, profundidade_rel)
    idade_final = int(idade_auto if idade_dias is None else np.clip(idade_dias, 0, 450))

    peso, conf_peso = estimar_peso(idade_final, area_relativa, perimetro_relativo, solidez, razao_bbox, profundidade_rel, raca)
    proje = gerar_projecao_peso_diaria(peso, idade_final, horizonte_dias=30)
    peso_min = float(proje["Peso mínimo (kg)"].iloc[0])
    peso_max = float(proje["Peso máximo (kg)"].iloc[0])
    ganho_dia = float(proje["Peso estimado (kg)"].iloc[1] - proje["Peso estimado (kg)"].iloc[0]) if len(proje) > 1 else 0.0

    return AnaliseBovino(
        raca_estimada=raca,
        idade_estimada_dias=idade_final,
        peso_estimado=round(peso, 1),
        peso_arroba=round(peso / 15.0, 2),
        faixa_min_kg=round(peso_min, 1),
        faixa_max_kg=round(peso_max, 1),
        ganho_dia_kg=round(ganho_dia, 3),
        confianca_raca=round(conf_raca, 2),
        confianca_idade=round(conf_idade, 2),
        confianca_peso=round(conf_peso, 2),
        area_relativa=round(area_relativa, 3),
        perimetro_relativo=round(perimetro_relativo, 3),
        solidez=round(solidez, 3),
        razao_bbox=round(razao_bbox, 3),
        profundidade_relativa=round(profundidade_rel, 3),
        backend_segmentacao=detectar_backends().get("backend_segmentacao", "OpenCV+GrabCut"),
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
