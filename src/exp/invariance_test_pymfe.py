#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np

from src.data.dataset import Dataset
from src.mtft import MetaFeatExtractor, MetaFeatures
from src.utils.logger import get_logger

# Опционально: PyMFE-вариант извлечения, если хочешь сравнивать именно PyMFE
USE_PYMFE = False
if USE_PYMFE:
    from pymfe.mfe import MFE

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def resolve_target(df, meta, explicit=None):
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
    raise ValueError(f"Cannot resolve target; first columns: {list(df.columns)[:8]}")

def extract_with_pymfe(ds: Dataset, groups=("general","statistical","info-theory"), summary=("mean","sd")) -> dict:
    tgt = resolve_target(ds.df, ds.meta)
    X = ds.df.drop(columns=[tgt]).to_numpy()
    y = ds.df[tgt].to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="mean").fit_transform(X)
    mfe = MFE(groups=groups, summary=summary, random_state=0)
    mfe.fit(X, y)
    names, vals = mfe.extract()
    out = {}
    for k, v in zip(names, vals):
        try:
            arr = np.asarray(v, dtype=float)
            out[f"pymfe__{k}"] = float(np.nanmean(arr)) if arr.ndim else float(arr)
        except Exception:
            out[f"pymfe__{k}"] = np.nan
    return out

def numeric_close(a, b, rtol, atol):
    try:
        fa = float(a)
        fb = float(b)
        return bool(np.isclose(fa, fb, rtol=rtol, atol=atol, equal_nan=True))
    except Exception:
        return a == b

def compare_value_dicts(d1: dict, d2: dict, rtol=1e-6, atol=1e-9):
    keys = set(d1.keys()) | set(d2.keys())
    diffs = {}
    for k in sorted(keys):
        v1 = d1.get(k, np.nan)
        v2 = d2.get(k, np.nan)
        if not numeric_close(v1, v2, rtol, atol):
            diffs[k] = (v1, v2)
    return len(diffs) == 0, diffs

def main():
    ap = argparse.ArgumentParser(description="Check invariance of meta-features under shuffling and column/category permutation.")
    ap.add_argument("csv", type=str, help="Path to dataset CSV, e.g., data/datasets/26.csv")
    ap.add_argument("meta", type=str, help="Path to dataset meta JSON, e.g., data/datasets/26.meta.json")
    ap.add_argument("--index", type=int, default=-1, help="Dataset index label for logs")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for float comparison")
    ap.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance for float comparison")
    ap.add_argument("--log", type=str, default="logs/exps/invariance_check.log", help="Log file path")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.log))
    logger = get_logger(log_file=args.log)

    # 1) Загружаем датасет
    ds = Dataset.load(args.csv, args.meta, args.index)
    logger.info(f"Loaded dataset index={ds.index}, shape={ds.df.shape}, columns={list(ds.df.columns)[:8]}...")

    # 2) Извлекаем признаки до перестановок
    if not USE_PYMFE:
        extractor = MetaFeatExtractor()
        mt_before = extractor.extract(ds)
        base_before = mt_before.base_feat
        stat_before = mt_before.stat_feat.values
        struct_before = mt_before.struct_feat.values
        logger.info("Extracted features (before shuffle) via project extractor.")
    else:
        # Если нужно PyMFE — кладем в “структурные” для сравнения словарей
        feats = extract_with_pymfe(ds)
        base_before = None
        stat_before = {}
        struct_before = feats
        logger.info("Extracted features (before shuffle) via PyMFE.")

    # 3) Копия датасета и перестановки (строки, столбцы, категории)
    ds_sh = Dataset.load(args.csv, args.meta, args.index)
    ds_sh.shuffle_dataset()
    logger.info(f"Shuffled dataset shape={ds_sh.df.shape}, columns={list(ds_sh.df.columns)[:8]}...")

    # 4) Извлекаем признаки после перестановок
    if not USE_PYMFE:
        mt_after = extractor.extract(ds_sh)
        base_after = mt_after.base_feat
        stat_after = mt_after.stat_feat.values
        struct_after = mt_after.struct_feat.values
        logger.info("Extracted features (after shuffle) via project extractor.")
    else:
        feats2 = extract_with_pymfe(ds_sh)
        base_after = None
        stat_after = {}
        struct_after = feats2
        logger.info("Extracted features (after shuffle) via PyMFE.")

    # 5) Сравнение
    if base_before is not None:
        eq_base, diff_base = MetaFeatures.compare_metafeat(base_before, base_after, rtol=args.rtol, atol=args.atol, nan_equal=True)
        logger.info(f"Base invariance: {eq_base}; diffs: {diff_base}")
    else:
        logger.info("Base invariance: skipped (PyMFE-only mode).")

    if stat_before is not None:
        eq_stat, diff_stat = compare_value_dicts(stat_before, stat_after, rtol=args.rtol, atol=args.atol)
        logger.info(f"Stat invariance: {eq_stat}; diffs_count={len(diff_stat)}")
        if not eq_stat:
            logger.info(f"Stat first diffs: {dict(list(diff_stat.items())[:10])}")
    else:
        logger.info("Stat invariance: skipped (PyMFE-only mode or empty).")

    eq_struct, diff_struct = compare_value_dicts(struct_before, struct_after, rtol=args.rtol, atol=args.atol)
    logger.info(f"Struct invariance: {eq_struct}; diffs_count={len(diff_struct)}")
    if not eq_struct:
        logger.info(f"Struct first diffs: {dict(list(diff_struct.items())[:10])}")

    # 6) Короткий итог в консоль
    print("Invariance check done. See log:", args.log)

if __name__ == "__main__":
    main()
