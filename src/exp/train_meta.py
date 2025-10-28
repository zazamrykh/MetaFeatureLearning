#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from src.utils.logger import get_logger
logger = get_logger(log_file='logs/exps/train_meta_pymfe.log')


def load_meta_and_quality(root_dir: str) -> pd.DataFrame:
    """
    Загружает i.json (метапризнаки) и i.txt (качества алгоритмов), собирает
    одну строку на датасет: все числовые метапризнаки + столбцы perf__algo + meta__best_algo.
    """
    rows = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        json_path = os.path.join(root_dir, f"{stem}.json")
        txt_path = os.path.join(root_dir, f"{stem}.txt")
        if not os.path.exists(txt_path):
            continue

        # метапризнаки
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)

        row = {}
        row["index"] = d.get("index", int(stem) if stem.isdigit() else -1)

        # base
        base = d.get("base_feat", {})
        row["base__n_samples"] = base.get("n_samples", np.nan)
        row["base__n_features"] = base.get("n_features", np.nan)
        row["base__n_cat"] = base.get("n_cat", np.nan)
        row["base__n_classes"] = base.get("n_classes", np.nan)

        # stat
        stat = d.get("stat_feat", {})
        stat_vals = stat.get("values", stat)
        if isinstance(stat_vals, dict):
            for k, v in stat_vals.items():
                row[f"stat__{k}"] = v

        # struct
        struct = d.get("struct_feat", {})
        struct_vals = struct.get("values", struct)
        if isinstance(struct_vals, dict):
            for k, v in struct_vals.items():
                row[f"struct__{k}"] = v

        # качества
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            perf = eval(content, {"__builtins__": {}})
        except Exception:
            perf = json.loads(content)

        # мета-метка: лучший алгоритм
        best_algo = max(perf.items(), key=lambda kv: kv[1])[0]
        row["meta__best_algo"] = str(best_algo)
        # сохраним и качества (опционально)
        for k, v in perf.items():
            row[f"perf__{k}"] = float(v)

        rows.append(row)

    return pd.DataFrame(rows)


def build_Xy(df: pd.DataFrame):
    """
    Формирует X, y для мета-обучения.
    Целевая переменная y = meta__best_algo (строка).
    Признаки — все числовые столбцы, кроме perf__* и служебных.
    """
    if df.empty:
        raise ValueError("No meta rows loaded")

    y = df["meta__best_algo"].astype(str)

    drop_cols = ["index", "meta__best_algo"]
    # Исключим сами perf__* столбцы из признаков, чтобы мета-обучение не было читерским
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and not c.startswith("perf__")
    ]

    # Оставим только числовые колонки
    num_cols = [c for c in feature_cols if df[c].dtype.kind in "if"]
    X = df[num_cols].copy()

    return X, y, num_cols


def fit_and_eval_models(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """
    Делит выборку на train/test, импутирует mean по колонкам, масштабирует,
    обучает 4 подхода: дерево, knn, логистическая регрессия, наивный DummyClassifier,
    и возвращает словарь с accuracy на тесте.
    """
    # Разбиение (страт. по мета-классу для баланса)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )  # [web:248]

    # Импутация средним: SimpleImputer(strategy="mean") [web:256]
    # Пайплайны
    pipe_tree = make_pipeline(
        SimpleImputer(strategy="mean"),
        DecisionTreeClassifier(max_depth=5, random_state=random_state),
    )
    pipe_knn = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(with_mean=True, with_std=True),
        KNeighborsClassifier(n_neighbors=5),
    )
    pipe_linear = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, n_jobs=1)
    )
    pipe_dummy = DummyClassifier(strategy="most_frequent")  # наивный базовый [web:260][web:254]

    # Обучение
    pipe_tree.fit(X_train, y_train)
    pipe_knn.fit(X_train, y_train)
    pipe_linear.fit(X_train, y_train)
    pipe_dummy.fit(X_train.fillna(X_train.mean()), y_train)  # для Dummy X не важен, но на всякий случай

    # Оценка
    preds = {
        "tree": pipe_tree.predict(X_test),
        "knn": pipe_knn.predict(X_test),
        "linear": pipe_linear.predict(X_test),
        "naive": pipe_dummy.predict(X_test.fillna(X_train.mean())),
    }
    acc = {name: accuracy_score(y_test, yhat) for name, yhat in preds.items()}
    return acc


def main():
    ap = argparse.ArgumentParser(description="Meta-learning: train/test models to predict best algorithm class.")
    ap.add_argument("indir", type=str, help="Directory with i.json (metafeatures) and i.txt (qualities)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random state")
    args = ap.parse_args()

    logger.info(f'Loading meta and quality from {args.indir}')
    df = load_meta_and_quality(args.indir)
    logger.info(f'Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns')

    X, y, cols = build_Xy(df)
    logger.info('Fit and eval models')
    scores = fit_and_eval_models(X, y, test_size=args.test_size, random_state=args.seed)

    logger.info("Test accuracy by approach:")
    for k, v in scores.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
