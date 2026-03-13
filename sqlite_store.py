from __future__ import annotations

import json
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


def _add_col_if_missing(conn: sqlite3.Connection, table: str, col: str, definition: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")


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
                backend_profundidade TEXT,
                projecao_json TEXT,
                deteccoes_json TEXT,
                imagem_original_b64 TEXT,
                imagem_final_b64 TEXT,
                mascara_final_b64 TEXT,
                idade_real INTEGER,
                raca_corrigida TEXT,
                peso_medido REAL,
                atualizado_em TEXT
            )
            """
        )

        # migração leve para bases antigas
        _add_col_if_missing(conn, "analises", "projecao_json", "TEXT")
        _add_col_if_missing(conn, "analises", "deteccoes_json", "TEXT")
        _add_col_if_missing(conn, "analises", "imagem_original_b64", "TEXT")
        _add_col_if_missing(conn, "analises", "imagem_final_b64", "TEXT")
        _add_col_if_missing(conn, "analises", "mascara_final_b64", "TEXT")
        _add_col_if_missing(conn, "analises", "idade_real", "INTEGER")
        _add_col_if_missing(conn, "analises", "raca_corrigida", "TEXT")
        _add_col_if_missing(conn, "analises", "peso_medido", "REAL")
        _add_col_if_missing(conn, "analises", "atualizado_em", "TEXT")


def import_dataset_if_empty() -> int:
    import csv

    init_db()
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) c FROM dataset_bovinos").fetchone()["c"]
        if count > 0 or not DATASET_PATH.exists():
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
            "INSERT INTO dataset_bovinos (id, tag, raca, idade_dias, peso_kg, origem) VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        return len(rows)


def create_analysis(resultado: Any, projection: list[dict[str, Any]], images: dict[str, str]) -> int:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO analises (
                raca_estimada, idade_estimada_dias, peso_estimado, peso_arroba,
                faixa_min_kg, faixa_max_kg, confianca_raca, confianca_idade,
                confianca_peso, backend_segmentacao, backend_profundidade,
                projecao_json, deteccoes_json, imagem_original_b64, imagem_final_b64, mascara_final_b64
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(projection, ensure_ascii=False),
                json.dumps(resultado.deteccoes_finais, ensure_ascii=False),
                images.get("original", ""),
                images.get("final", ""),
                images.get("mask", ""),
            ),
        )
        return int(cur.lastrowid)


def read_analyses(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM analises ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def read_analysis(analysis_id: int) -> dict[str, Any] | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM analises WHERE id = ?", (analysis_id,)).fetchone()
        return dict(row) if row else None


def update_analysis(analysis_id: int, idade_real: int | None, raca_corrigida: str | None, peso_medido: float | None) -> bool:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE analises
            SET idade_real = ?, raca_corrigida = ?, peso_medido = ?, atualizado_em = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (idade_real, raca_corrigida, peso_medido, analysis_id),
        )
        return cur.rowcount > 0


def delete_analysis(analysis_id: int) -> bool:
    init_db()
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM analises WHERE id = ?", (analysis_id,))
        return cur.rowcount > 0


def list_dataset_bovinos(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM dataset_bovinos ORDER BY id ASC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
