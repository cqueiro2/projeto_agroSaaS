from dataclasses import dataclass

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
    mascara: np.ndarray
    imagem_segmentada: np.ndarray


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


def segmentar_bovino(imagem_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    imagem_small, escala = _normalizar_imagem(imagem_bgr)
    h, w = imagem_small.shape[:2]

    suavizada = cv2.bilateralFilter(imagem_small, d=9, sigmaColor=55, sigmaSpace=55)
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mascara_cor = cv2.inRange(hsv, (0, 25, 25), (179, 255, 245))
    mascara_textura = cv2.inRange(sat, 30, 255)
    mascara_luz = cv2.inRange(val, 20, 245)
    pre_fg = cv2.bitwise_and(mascara_cor, cv2.bitwise_and(mascara_textura, mascara_luz))

    # Priori central para reduzir captação de fundo
    centro = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(centro, (w // 2, h // 2), (int(w * 0.40), int(h * 0.40)), 0, 0, 360, 255, -1)
    pre_fg = cv2.bitwise_and(pre_fg, centro)

    kernel = np.ones((5, 5), np.uint8)
    pre_fg = cv2.morphologyEx(pre_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    pre_fg = cv2.morphologyEx(pre_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[pre_fg > 0] = cv2.GC_PR_FGD

    # Moldura externa como fundo certo
    borda = max(8, int(min(h, w) * 0.06))
    gc_mask[:borda, :] = cv2.GC_BGD
    gc_mask[-borda:, :] = cv2.GC_BGD
    gc_mask[:, :borda] = cv2.GC_BGD
    gc_mask[:, -borda:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(suavizada, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass

    mask_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
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

    # Volta para resolução original para visualização no app
    if escala != 1.0:
        h0, w0 = imagem_bgr.shape[:2]
        mask_fg = cv2.resize(mask_fg, (w0, h0), interpolation=cv2.INTER_NEAREST)
        segmentada = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mask_fg)

    area_relativa = float(np.count_nonzero(mask_fg) / (mask_fg.shape[0] * mask_fg.shape[1]))
    perimetro_relativo = float(perimetro / (2 * (h + w)))
    return mask_fg, segmentada, area_relativa, perimetro_relativo, solidez, razao_bbox


def estimar_raca(imagem_segmentada: np.ndarray, mascara: np.ndarray) -> tuple[str, float]:
    pixels = imagem_segmentada[mascara == 255]
    if len(pixels) < 300:
        return "Indefinida", 0.35

    hsv = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2HSV)[0]
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    branco = float(np.mean((v > 170) & (s < 70)))
    escuro = float(np.mean(v < 70))
    avermelhado = float(np.mean(((h >= 5) & (h <= 25) & (s > 70) & (v > 60))))
    misto = float(np.mean((v > 90) & (v < 170) & (s > 45)))

    scores = {
        "Nelore": 0.30 + 0.95 * branco - 0.35 * escuro,
        "Angus": 0.30 + 1.05 * escuro - 0.30 * branco,
        "Jersey": 0.28 + 1.05 * avermelhado,
        "Girolando": 0.30 + 0.70 * misto + 0.30 * branco,
    }

    raca = max(scores, key=scores.get)
    ordenado = sorted(scores.values(), reverse=True)
    margem = max(0.02, ordenado[0] - ordenado[1])
    confianca = float(np.clip(0.60 + margem * 0.85, 0.50, 0.90))
    return raca, confianca


def estimar_peso(
    idade_dias: int,
    area_relativa: float,
    perimetro_relativo: float,
    solidez: float,
    razao_bbox: float,
    raca: str,
) -> tuple[float, float]:
    idade = int(np.clip(idade_dias, 0, 450))

    # Curva de crescimento (Gompertz simplificada)
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
    fator_forma += (area_relativa - 0.27) * 0.95
    fator_forma += (solidez - 0.82) * 0.55
    fator_forma += (razao_bbox - 1.45) * 0.10
    fator_forma += (perimetro_relativo - 0.20) * 0.35
    fator_forma = float(np.clip(fator_forma, 0.72, 1.28))

    peso = float(peso_base * fator_forma)

    qualidade = 1.0
    qualidade -= min(abs(area_relativa - 0.30), 0.30)
    qualidade -= min(abs(solidez - 0.85), 0.30) * 0.7
    qualidade -= min(abs(razao_bbox - 1.50), 0.80) * 0.15

    confianca = float(np.clip(0.62 + max(0.0, qualidade) * 0.28, 0.55, 0.90))
    return peso, confianca


def analisar_bovino(imagem_bgr: np.ndarray, idade_dias: int) -> AnaliseBovino:
    mascara, segmentada, area_relativa, perimetro_relativo, solidez, razao_bbox = segmentar_bovino(imagem_bgr)
    raca, conf_raca = estimar_raca(segmentada, mascara)
    peso, conf_peso = estimar_peso(idade_dias, area_relativa, perimetro_relativo, solidez, razao_bbox, raca)

    return AnaliseBovino(
        peso_estimado=round(peso, 1),
        raca_estimada=raca,
        confianca_peso=round(conf_peso, 2),
        confianca_raca=round(conf_raca, 2),
        area_relativa=round(area_relativa, 3),
        perimetro_relativo=round(perimetro_relativo, 3),
        solidez=round(solidez, 3),
        razao_bbox=round(razao_bbox, 3),
        mascara=mascara,
        imagem_segmentada=segmentada,
    )


def gerar_curva_crescimento(peso_atual: float, idade_dias: int) -> pd.DataFrame:
    dias = np.arange(0, 451, 15)
    idade_ref = max(idade_dias, 1)
    fator = np.clip((dias + 1) / (idade_ref + 1), 0.15, 1.25)
    pesos = np.clip(peso_atual * (fator ** 0.88), 30, 460)
    return pd.DataFrame({"Idade (dias)": dias, "Peso estimado (kg)": pesos})
