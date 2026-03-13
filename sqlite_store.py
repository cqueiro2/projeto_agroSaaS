import csv
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/bovino.db")
DATASET_PATH = Path("data/bovinos_dataset.csv")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_bovinos (
                id INTEGER PRIMARY KEY,
                tag TEXT,
                raca TEXT,
                idade_dias INTEGER,
                peso_kg REAL,
                origem TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                criado_em TEXT DEFAULT CURRENT_TIMESTAMP,
                raca_estimada TEXT,
                idade_estimada_dias INTEGER,
                peso_estimado REAL,
                peso_arroba REAL,
                faixa_min_kg REAL,
                faixa_max_kg REAL,
                confianca_raca REAL,
                confianca_idade REAL,
                confianca_peso REAL,
                backend_segmentacao TEXT,
                backend_profundidade TEXT
            )
            """
        )


def import_dataset_if_empty() -> int:
    init_db()
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM dataset_bovinos").fetchone()["c"]
        if count > 0:
            return 0
        if not DATASET_PATH.exists():
            return 0
        with DATASET_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                (
                    int(r["id"]),
                    r["tag"],
                    r["raca"],
                    int(r["idade_dias"]),
                    float(r["peso_kg"]),
                    r["origem"],
                )
                for r in reader
            ]
        conn.executemany(
            "INSERT INTO dataset_bovinos (id, tag, raca, idade_dias, peso_kg, origem) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)


def save_analysis(resultado: Any) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO analises (
                raca_estimada, idade_estimada_dias, peso_estimado, peso_arroba,
                faixa_min_kg, faixa_max_kg, confianca_raca, confianca_idade,
                confianca_peso, backend_segmentacao, backend_profundidade
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resultado.raca_estimada,
                int(resultado.idade_estimada_dias),
                float(resultado.peso_estimado),
                float(resultado.peso_arroba),
                float(resultado.faixa_min_kg),
                float(resultado.faixa_max_kg),
                float(resultado.confianca_raca),
                float(resultado.confianca_idade),
                float(resultado.confianca_peso),
                resultado.backend_segmentacao,
                resultado.backend_profundidade,
            ),
        )


def list_recent_analyses(limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analises ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
