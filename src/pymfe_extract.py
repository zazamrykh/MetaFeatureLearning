#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import simplejson as sjson
from pymfe.mfe import MFE
from typing import Optional

# Настройки
IN_DIR = "data/datasets"
OUT_DIR = "data/pymfe_metafeatures"
os.makedirs(OUT_DIR, exist_ok=True)

def resolve_target(df: pd.DataFrame, meta: dict, explicit: Optional[str] = None) -> str:
    cands = []
    if explicit:
        cands.append(explicit)
    cands += [
        meta.get("target_used"),
        meta.get("default_target_attribute"),
        "target", "class", "label", "y", "binaryClass",
    ]
    cands = [c for c in cands if isinstance(c, str)]
    for c in cands:
        if c in df.columns:
            return c
    non_num = [c for c in df.columns if df[c].dtype.kind not in "if"]
    if len(non_num) == 1:
        return non_num[0]
    raise ValueError(f"Cannot resolve target. Columns head: {list(df.columns)[:8]}")

def extract_pymfe_for_file(stem: str):
    df_path = os.path.join(IN_DIR, f"{stem}.csv")
    meta_path = os.path.join(IN_DIR, f"{stem}.meta.json")
    if not (os.path.exists(df_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(df_path)

    # Таргет
    target = resolve_target(df, meta)
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()

    # Очистка и иммутация
    X = np.where(np.isfinite(X), X, np.nan)
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="mean").fit_transform(X)

    # PyMFE: базовый стабильный набор
    mfe = MFE(groups=("general", "statistical", "info-theory"), summary=("mean", "sd"), random_state=0)
    mfe.fit(X, y)
    names, vals = mfe.extract()

    feats = {}
    for name, val in zip(names, vals):
        try:
            arr = np.asarray(val, dtype=float)
            feats[f"pymfe__{name}"] = float(np.nanmean(arr)) if arr.ndim else float(arr)
        except Exception:
            feats[f"pymfe__{name}"] = None

    # Собираем JSON в твоём формате (base/stat пустые или базовые)
    out = {
        "index": int(stem) if stem.isdigit() else stem,
        "base_feat": {
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1),
            "n_cat": int((df.drop(columns=[target]).dtypes.map(lambda t: t.kind not in "if")).sum()),
            "n_classes": int(meta.get("qualities", {}).get("NumberOfClasses", 0)),
        },
        "stat_feat": {
            "values": {}  # можно оставить пустым или заполнить твоими расчетами
        },
        "struct_feat": {
            "values": feats  # кладём PyMFE сюда, чтобы формат был совместим
        },
    }
    return out

def main():
    saved = 0
    for f in os.listdir(IN_DIR):
        if not f.endswith(".csv"):
            continue
        stem = f[:-4]
        try:
            obj = extract_pymfe_for_file(stem)
        except Exception as e:
            print(f"[WARN] skip {stem}: {e}")
            continue
        if obj is None:
            continue
        out_path = os.path.join(OUT_DIR, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            sjson.dump(obj, fh, ensure_ascii=False, indent=2, ignore_nan=True)
        saved += 1
    print(f"Saved {saved} files to {OUT_DIR}")

if __name__ == "__main__":
    main()
