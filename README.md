# 🌾 AgroSaaS

O **AgroSaaS** é uma plataforma voltada para o agronegócio, desenvolvida para facilitar a gestão, análise e automação de processos agrícolas.

## 🚀 Módulo Bovino (Flask + interface responsiva)

Este repositório inclui uma aplicação **Flask** com layout responsivo (mobile-first) para:
- 📷 Receber foto do bovino.
- ✂️ Detectar bovinos com saída limpa (somente resultado final: bbox/rótulo/score da classe bovino).
- ⚖️ Estimar raça, idade (dias), peso (kg e arrobas) e faixa de variação.
- 📈 Gerar projeção diária de peso e curva de crescimento.
- 🧠 YOLO com filtro de classe bovino e fallback heurístico, ocultando visualizações intermediárias.
- 🗄️ Persistir análises em SQLite e carregar dataset inicial automaticamente.

> A confiança das estimativas é limitada a até 90% e não substitui avaliação zootécnica/pesagem real.

## ▶️ Como executar (Flask)

```bash
pip install -r requirement.txt
python app_flask.py
```

Acesse: `http://localhost:5000`

## 📁 Estrutura principal

- `app_flask.py`: aplicação Flask (backend + rotas).
- `templates/index.html`: interface responsiva.
- `static/styles.css`: estilo mobile-first da aplicação.
- `bovino_analise.py`: motor de visão computacional e estimativas.
- `sqlite_store.py`: inicialização SQLite, importação de dataset e histórico.
- `data/bovinos_dataset.csv`: dataset inicial de bovinos para popular o banco.
- `data/bovino.db`: banco SQLite criado automaticamente em runtime.
- `app_bovino.py`: versão Streamlit mantida para compatibilidade.

## ⚠️ Aviso técnico

As predições de peso, idade e raça são **estimativas heurísticas** baseadas em imagem.
Para decisão operacional, utilize pesagem em balança e validação de um profissional.


### 🎬 Vídeo quadro a quadro
Use `processar_frame_video_final(frame_bgr)` em `bovino_analise.py` para renderizar apenas o resultado final em cada frame.
