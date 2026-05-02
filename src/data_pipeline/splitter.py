"""
Split train/val/test des datasets anonymisés.
Stratifié sur source + language. eval_clinique prélevé en premier.
Déduplication MinHash (Jaccard ≥ 0.9) avant le split.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

from datasketch import MinHash, MinHashLSH


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for w in text.lower().split():
        m.update(w.encode("utf-8"))
    return m


def _deduplicate(records: list[dict], threshold: float = 0.9, num_perm: int = 128) -> list[dict]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for r in records:
        m = _minhash(r["instruction"], num_perm)
        if not lsh.query(m):
            lsh.insert(r["id"], m)
            kept.append(r)
    return kept


def split_sft(anon_dir: Path, output_dir: Path, eval_dir: Path,
              train_ratio: float, val_ratio: float, seed: int,
              eval_size: int) -> dict:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Charger tous les fichiers SFT anonymisés
    all_records = []
    for f in anon_dir.glob("*_anonymized.jsonl"):
        if "dpo" in f.name:
            continue
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                all_records.append(json.loads(line))

    random.shuffle(all_records)

    # Déduplication MinHash avant split
    before = len(all_records)
    all_records = _deduplicate(all_records)
    print(f"  Déduplication SFT : {before} → {len(all_records)} ({before - len(all_records)} supprimés)")

    # Extraire eval_clinique en premier (100 premiers après shuffle)
    eval_records = all_records[:eval_size]
    remaining = all_records[eval_size:]

    # Stratification par source
    by_source = defaultdict(list)
    for r in remaining:
        by_source[r["source"]].append(r)

    train_records, val_records, test_records = [], [], []
    for source, records in by_source.items():
        random.shuffle(records)
        n = len(records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_records.extend(records[:n_train])
        val_records.extend(records[n_train:n_train + n_val])
        test_records.extend(records[n_train + n_val:])

    # Assigner le champ split et écrire
    def write_split(records, path, split_name):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                r["metadata"]["split"] = split_name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_split(eval_records, eval_dir / "eval_clinique.jsonl", "eval")
    write_split(train_records, output_dir / "train.jsonl", "train")
    write_split(val_records, output_dir / "val.jsonl", "val")
    write_split(test_records, output_dir / "test.jsonl", "test")

    return {
        "eval": len(eval_records),
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
    }


def split_dpo(anon_dir: Path, output_dir: Path,
              train_ratio: float, val_ratio: float, seed: int) -> dict:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for f in anon_dir.glob("*dpo*_anonymized.jsonl"):
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                all_records.append(json.loads(line))

    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train:n_train + n_val],
        "test": all_records[n_train + n_val:],
    }

    for split_name, records in splits.items():
        with open(output_dir / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                r["metadata"]["split"] = split_name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {k: len(v) for k, v in splits.items()}
