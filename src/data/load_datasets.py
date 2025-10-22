#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import pandas as pd

# openml-python API
# Docs and examples: https://openml.github.io/openml-python/ (APIs to list and get datasets)
import openml  # noqa: F401

# For robust typing when extracting data
from openml.datasets import get_dataset

"""
Загрузка N датасетов с OpenML для задач классификации и сохранение их в удобном формате.

Сохраняется:
- <k>.csv — табличные данные с целевой колонкой target
- <k>.meta.json — метаданные датасета (id, имя, размеры, число классов, target_name, и т.д.)

Критерии фильтрации управляются флагами CLI.

Примеры запуска:
python3 src/data/load_datasets.py --output-dir data/raw/openml --n-datasets 300
python3 src/data/load_datasets.py --output-dir data/raw/openml --n-datasets 300 --max-instances 20000 --max-features 500
python3 src/data/load_datasets.py --output-dir data/raw/openml --n-datasets 300 --tags uci
"""

DEFAULT_TARGET_COL = "target"


def parse_args():
    p = argparse.ArgumentParser(description="Download multiple OpenML datasets for classification and save as numbered CSV + JSON meta.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save datasets (will be created if not exists).")
    p.add_argument("--n-datasets", type=int, default=300, help="Number of datasets to download and save.")
    p.add_argument("--min-instances", type=int, default=300, help="Minimum number of instances.")
    p.add_argument("--max-instances", type=int, default=3000, help="Maximum number of instances to allow.")
    p.add_argument("--max-features", type=int, default=100, help="Maximum number of features to allow.")
    p.add_argument("--max-classes", type=int, default=30, help="Maximum number of classes to allow.")
    p.add_argument("--require_default_target", action="store_true", help="Only datasets with a default target attribute.")
    p.add_argument("--tags", type=str, nargs="*", default=None, help="Optional OpenML tags to filter datasets (e.g., uci).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for shuffling selection order.")
    p.add_argument("--sleep-sec", type=float, default=0.5, help="Sleep between downloads to be polite to the API.")
    p.add_argument("--use_tasks", action="store_true", help="Prefer sampling via OpenML tasks for classification instead of raw datasets.")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_candidate_datasets(
    min_instances: int,
    max_instances: int,
    max_features: int,
    max_classes: int,
    require_default_target: bool,
    tags: Optional[List[str]],
    use_tasks: bool,
    random_state: int,
) -> pd.DataFrame:
    """
    Возвращает датафрейм с кандидатами на загрузку.

    Стратегии:
    - Если use_tasks=True: берем classification tasks и их датасеты (часто более надёжно для supervised). 
    - Иначе: напрямую list_datasets с фильтрами по качествам (instances/features/classes).
    """
    rng = pd.Series(range(1)).sample(frac=1.0, random_state=random_state)  # just to consume seed

    if use_tasks:
        # Получаем задачи классификации и затем вытягиваем уникальные dataset ids.
        # Руководство и примеры API см. в доках openml. 
        # После этого для каждого did проверяем качества. [OpenML user guide]
        task_list = openml.tasks.list_tasks(output_format="dataframe")  # may be big; you could filter type
        # Filter classification tasks only
        task_list = task_list[task_list["task_type"] == "Supervised Classification"]
        # map to dataset ids
        dids = task_list["did"].dropna().astype(int).unique()
        # Build a small DataFrame to probe qualities via datasets list
        ds_df = openml.datasets.list_datasets(output_format="dataframe")
        df = ds_df[ds_df["did"].isin(dids)].copy()
    else:
        # Прямой список датасетов c качествами: NumberOfInstances, NumberOfFeatures, NumberOfClasses. [datasets tutorial]
        df = openml.datasets.list_datasets(output_format="dataframe")

    # Базовые фильтры размеров
    df = df[
        (df["NumberOfInstances"].fillna(0) >= min_instances)
        & (df["NumberOfInstances"].fillna(0) <= max_instances)
        & (df["NumberOfFeatures"].fillna(0) > 1)
        & (df["NumberOfFeatures"].fillna(0) <= max_features)
    ]

    # Классы: допускаем те, у кого есть NumberOfClasses и он не слишком большой
    if "NumberOfClasses" in df.columns:
        df = df[df["NumberOfClasses"].fillna(0) <= max_classes]

    # Только активные
    if "status" in df.columns:
        df = df[df["status"] == "active"]

    # Только табличные форматы (ARFF/CSV/Parquet/…); чаще ARFF/CSV
    if "format" in df.columns:
        df = df[df["format"].isin(["ARFF", "CSV", "Parquet"])]

    # Фильтр по тегам, если задан
    if tags:
        # У datasets list_datasets может не быть прямого столбца с тегами; 
        # поэтому фильтрацию по тегам надёжнее делать через openml.search, 
        # но для простоты оставим проброс и попытаемся матчингу по 'tag' если есть:
        tag_col = "tag"
        if tag_col in df.columns:
            mask = pd.Series(False, index=df.index)
            for t in tags:
                mask = mask | df[tag_col].fillna("").astype(str).str.contains(t, case=False)
            df = df[mask]

    # Требовать наличие default_target_attribute
    if require_default_target and "default_target_attribute" in df.columns:
        df = df[df["default_target_attribute"].notna() & (df["default_target_attribute"].astype(str).str.len() > 0)]

    # Случайно перемешаем для разнообразия
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Оставим полезные колонки
    keep_cols = [
        "did",
        "name",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfClasses",
        "default_target_attribute",
        "format",
        "status",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]


def dataset_to_frame(dataset_obj, target_name: Optional[str]):
    """
    Возвращает (df, meta) для сохранения. df содержит признаки и target.
    """
    # Если целевая не задана, пробуем дефолтную из объекта, иначе используем DEFAULT_TARGET_COL позже
    if target_name is None:
        target_name = dataset_obj.default_target_attribute

    # Извлекаем данные. По API openml-python:
    # X, y, categorical_indicator, attribute_names = dataset.get_data(target=target_name)
    # [see openml minimal example and dataset tutorial]
    X, y, cat_ind, attr_names = dataset_obj.get_data(target=target_name)

    # Сформируем DataFrame
    X = pd.DataFrame(X, columns=[str(a) for a in attr_names])
    if y is not None:
        y_series = pd.Series(y, name="target")
    else:
        # Если target не было, создадим пустой столбец, но лучше такие датасеты пропустить выше
        y_series = pd.Series([None] * len(X), name="target")

    df = pd.concat([X, y_series], axis=1)

    meta = {
        "did": dataset_obj.dataset_id,
        "name": dataset_obj.name,
        "version": getattr(dataset_obj, "version", None),
        "url": getattr(dataset_obj, "url", None),
        "description_present": bool(getattr(dataset_obj, "description", "") is not None),
        "default_target_attribute": dataset_obj.default_target_attribute,
        "target_used": target_name,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "categorical_indicator": list(map(bool, cat_ind)) if cat_ind is not None else None,
        "attribute_names": [str(a) for a in attr_names],
    }
    return df, meta


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    candidates = list_candidate_datasets(
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        max_features=args.max_features,
        max_classes=args.max_classes,
        require_default_target=args.require_default_target,
        tags=args.tags,
        use_tasks=args.use_tasks,
        random_state=args.random_state,
    )

    if len(candidates) == 0:
        print("No candidate datasets matched the filters.", file=sys.stderr)
        sys.exit(2)

    saved = 0
    attempted = 0
    idx = 0

    # Чтобы гарантировать нумерацию 1..N по факту сохранения
    while saved < args.n_datasets and idx < len(candidates):
        row = candidates.iloc[idx]
        idx += 1
        attempted += 1

        did = int(row["did"])
        name = str(row.get("name", f"did_{did}"))
        target_attr = row.get("default_target_attribute", None)
        print(f"[{attempted}] Fetching did={did} name={name} target={target_attr}")

        try:
            # Начиная с 0.15, ленивые загрузки; явно укажем download_data=True, qualities=False
            dataset = get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)  # see docs examples

            df, meta = dataset_to_frame(dataset, target_attr if pd.notna(target_attr) and str(target_attr) != "" else None)

            # Базовые проверки
            if "target" not in df.columns:
                print(f"  -> No target extracted; skipping did={did}")
                time.sleep(args.sleep_sec)
                continue

            # Уберём строки с полностью пустыми значениями target
            if df["target"].isna().all():
                print(f"  -> All targets are NA; skipping did={did}")
                time.sleep(args.sleep_sec)
                continue

            # Сохраняем
            k = saved + 1
            csv_path = os.path.join(args.output_dir, f"{k}.csv")
            meta_path = os.path.join(args.output_dir, f"{k}.meta.json")

            # Конвертируем категориальные метки в строковый вид для стабильности
            # и избегания неоднозначной типизации
            df = df.copy()
            if df["target"].dtype.name not in ("int64", "float64", "bool"):
                df["target"] = df["target"].astype(str)

            df.to_csv(csv_path, index=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": "openml",
                        "did": int(meta["did"]) if meta["did"] is not None else None,
                        "name": meta["name"],
                        "version": meta["version"],
                        "url": meta["url"],
                        "default_target_attribute": meta["default_target_attribute"],
                        "target_used": meta["target_used"],
                        "n_rows": meta["n_rows"],
                        "n_features": meta["n_features"],
                        "categorical_indicator": meta["categorical_indicator"],
                        "attribute_names": meta["attribute_names"],
                        "qualities": {
                            "NumberOfInstances": int(row.get("NumberOfInstances", -1)) if pd.notna(row.get("NumberOfInstances", None)) else None,
                            "NumberOfFeatures": int(row.get("NumberOfFeatures", -1)) if pd.notna(row.get("NumberOfFeatures", None)) else None,
                            "NumberOfClasses": int(row.get("NumberOfClasses", -1)) if pd.notna(row.get("NumberOfClasses", None)) else None,
                            "format": row.get("format", None),
                            "status": row.get("status", None),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            saved += 1
            print(f"  -> Saved {csv_path} and {meta_path} (did={did})")
        except Exception as e:
            print(f"  -> Error on did={did}: {e}; skipping.")
        finally:
            time.sleep(args.sleep_sec)

    print(f"Finished: saved={saved} requested={args.n_datasets} candidates_scanned={idx}")


if __name__ == "__main__":
    main()
