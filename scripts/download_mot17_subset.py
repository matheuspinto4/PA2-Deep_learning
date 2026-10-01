"""Baixa só o subconjunto do MOT17 que vamos usar, via range-requests HTTP
(o servidor do motchallenge.net suporta Accept-Ranges, confirmado com
`curl -I`), usando a lib `remotezip` para ler só as entradas específicas do
MOT17.zip (5.5GB) sem baixar o arquivo inteiro.

Sequências escolhidas e por quê (câmera parada vs. móvel, densidade,
ver AI_LOG.md para a tabela completa de densidade por sequência):

  - MOT17-02 (treino):      estática,  31.0 ped/quadro, 600 quadros
  - MOT17-09 (validação):   estática,  10.1 ped/quadro, 525 quadros
  - MOT17-04 (diagnóstico): estática,  45.3 ped/quadro, 1050 quadros (só
    entra no gráfico de dificuldade do item 5 da Parte 1, não é usada para
    treinar nem para decidir hiperparâmetro nenhum)
  - MOT17-11 (teste, held-out): câmera em MOVIMENTO, 10.5 ped/quadro, 900
    quadros -- nunca tocada durante treino/tuning. Critério de split: tipo
    de movimento de câmera, exatamente o eixo que o enunciado sugere.

Para cada sequência, baixamos as imagens (img1/) e o gt.txt de uma única
variante (FRCNN -- imagens e gt são idênticos entre as 3 variantes de
detector de uma mesma sequência física), mas o det.txt das 3 variantes
(DPM/FRCNN/SDP), já que são arquivos de texto pequenos e precisamos comparar
os 3 para justificar a escolha do detector público padrão (Parte 1, item 1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "MOT17"
ZIP_URL = "https://motchallenge.net/data/MOT17.zip"

SEQUENCES = {
    "02": "train",
    "09": "val",
    "04": "diagnostic",
    "11": "test",
}
DET_VARIANTS = ["DPM", "FRCNN", "SDP"]
IMG_GT_VARIANT = "FRCNN"


def remote_entries_for(seq_num: str):
    base = f"MOT17/train/MOT17-{seq_num}"
    entries = [f"{base}-{IMG_GT_VARIANT}/seqinfo.ini", f"{base}-{IMG_GT_VARIANT}/gt/gt.txt"]
    for variant in DET_VARIANTS:
        entries.append(f"{base}-{variant}/det/det.txt")
    return entries, base


def fetch_with_retry(fn, tries=4, delay=3):
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # conexão longa pode cair; reabrir do zero
            last_err = e
            print(f"    retry {attempt + 1}/{tries} após erro: {e}")
            time.sleep(delay)
    raise last_err


def download_sequence(seq_num: str, split: str, with_images: bool = True):
    dest = DATA_DIR / split / f"MOT17-{seq_num}"
    (dest / "det").mkdir(parents=True, exist_ok=True)
    (dest / "gt").mkdir(parents=True, exist_ok=True)

    entries, base = remote_entries_for(seq_num)

    print(f"[{seq_num}/{split}] baixando seqinfo.ini, gt.txt, det.txt (3 variantes)...")

    def do_text():
        with RemoteZip(ZIP_URL) as z:
            data_seqinfo = z.read(f"{base}-{IMG_GT_VARIANT}/seqinfo.ini")
            data_gt = z.read(f"{base}-{IMG_GT_VARIANT}/gt/gt.txt")
            data_dets = {v: z.read(f"{base}-{v}/det/det.txt") for v in DET_VARIANTS}
            return data_seqinfo, data_gt, data_dets

    data_seqinfo, data_gt, data_dets = fetch_with_retry(do_text)

    (dest / "seqinfo.ini").write_bytes(data_seqinfo)
    (dest / "gt" / "gt.txt").write_bytes(data_gt)
    for v, content in data_dets.items():
        (dest / "det" / f"det_{v}.txt").write_bytes(content)

    if with_images:
        img_dest = dest / "img1"
        img_dest.mkdir(exist_ok=True)
        existing = {p.name for p in img_dest.glob("*.jpg")}

        def list_images():
            with RemoteZip(ZIP_URL) as z:
                return [n for n in z.namelist() if n.startswith(f"{base}-{IMG_GT_VARIANT}/img1/") and n.endswith(".jpg")]

        names = fetch_with_retry(list_images)
        todo = [n for n in names if Path(n).name not in existing]
        print(f"[{seq_num}/{split}] {len(names)} imagens no total, {len(todo)} faltando, baixando em lotes...")

        batch_size = 60
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]

            def do_batch(batch=batch):
                with RemoteZip(ZIP_URL) as z:
                    for name in batch:
                        data = z.read(name)
                        (img_dest / Path(name).name).write_bytes(data)

            fetch_with_retry(do_batch)
            print(f"    {min(i + batch_size, len(todo))}/{len(todo)}")

    print(f"[{seq_num}/{split}] OK -> {dest}")


if __name__ == "__main__":
    only_text = "--text-only" in sys.argv
    for seq_num, split in SEQUENCES.items():
        download_sequence(seq_num, split, with_images=not only_text)
    print("\nFeito. Dados em", DATA_DIR)
