"""Comando único de avaliação (tabela de entregáveis do enunciado: "um
comando que avalia"). Roda a avaliação central da Parte 2: MotionGRU vs.
tracker ingênuo da Parte 1, mesmas métricas (mAP, IDF1, ID switches,
fragmentações, erro de contagem), mesmas 4 sequências, usando o checkpoint
final já treinado (`checkpoints/motion_gru_mot17.pt`) -- sem retreinar.

As demais análises (ablação da Parte 3, galeria de falhas e horizonte de
memória da Parte 4, teste de estresse da Parte 5) são leituras adicionais
em cima do mesmo checkpoint, não uma segunda forma de "avaliar o modelo";
ver os comandos de cada uma na seção "Comandos" do README.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable, str(ROOT / "part2_evaluate.py")], check=True)


if __name__ == "__main__":
    main()
