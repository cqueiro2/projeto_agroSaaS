# 🌾 AgroSaaS

O **AgroSaaS** é uma plataforma voltada para o agronegócio, desenvolvida para facilitar a gestão, análise e automação de processos agrícolas.

## 🚀 Módulo Bovino (imagem + idade até 450 dias)

Este repositório agora inclui um aplicativo em **Streamlit** para:
- 📷 Receber foto do bezerro/bovino.
- ✂️ Segmentar automaticamente o corpo do animal (cor + profundidade + GrabCut).
- ⚖️ Estimar automaticamente idade (dias) e peso atual com base em visão computacional + profundidade.
- 🐄 Sugerir raça provável (heurística) com confiança.
- 📈 Exibir curva estimada de crescimento até 450 dias.
- 📆 Projetar ganho de peso diário com faixa mínima/máxima para os próximos dias.
- 🧠 Detectar backend disponível (PyTorch/MiDaS e YOLO opcional) com fallback automático.

> A confiança do sistema é intencionalmente limitada em até **90%** para refletir estimativa realista e não substituir avaliação zootécnica profissional.

## ▶️ Como executar

```bash
pip install -r requirement.txt
streamlit run app_bovino.py
# ou, para compatibilidade:
python bovino_anlise.py
```

Abra o endereço exibido no terminal (normalmente `http://localhost:8501`).

> Execução recomendada: `streamlit run app_bovino.py`.

## 📁 Estrutura principal

- `app_bovino.py`: interface web Streamlit para análise bovina.
- `bovino_analise.py`: motor de visão computacional (segmentação + peso + raça).
- `1_DimensioImag/`: scripts de redimensionamento de imagens.
- `2_depthImag/`: experimentos com mapa de profundidade.
- `3_segmentarImg/`: scripts de segmentação de imagem.

## ⚠️ Aviso técnico

As predições de peso e raça são **estimativas heurísticas** baseadas em imagem e idade informada.
Para decisão operacional, utilize pesagem em balança e validação de um profissional.
