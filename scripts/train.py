"""Comando único de treino (tabela de entregáveis do enunciado: "um comando
que treina"). Roda a Parte 2 (Trilha A) em sequência -- pré-treino no
sintético, depois treino no MOT17 real -- e produz
`checkpoints/motion_gru_mot17.pt`, o checkpoint final usado por todas as
partes seguintes (3, 4, 5) e por `notebooks/inferencia.ipynb`.

Pra rodar só uma etapa, ou pra reproduzir a ablação da Parte 3 (que treina
as variantes RNN/LSTM/GRU separadamente), ver a lista completa de comandos
na seção "Comandos" do README.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = ["part2_pretrain_synthetic.py", "part2_train_mot17.py"]


def main():
    for step in STEPS:
        print(f"\n=== {step} ===")
        subprocess.run([sys.executable, str(ROOT / step)], check=True)


if __name__ == "__main__":
    main()
