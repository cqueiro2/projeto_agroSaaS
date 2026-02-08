"""Compatibilidade de nome de arquivo para abrir o dashboard.

Uso:
    python bovino_anlise.py

Este script inicia o Streamlit apontando para `app_bovino.py`.
"""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    app_path = Path(__file__).with_name("app_bovino.py")
    if not app_path.exists():
        print("Erro: app_bovino.py não encontrado no mesmo diretório.")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
