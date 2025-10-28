#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import simplejson as json
import argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer  # для mean-импутации [web:256]

# Импортируйте ваши классы, как просили
from src.mtft import MetaFeatures, BaseMetaFeat, StatisticMetaFeat, StructuredMetaFeat
from src.data.dataset import Dataset  # контейнер
from src.utils.logger import get_logger
logger = get_logger()


def load_meta_rows(indir: str) -> pd.DataFrame:
    """
    Загружает пары i.json / i.txt, строит плоские строки признаков и добавляет колонку meta__best_algo.
    Использует MetaFeatures.load_from_json и .to_dataframe.
    """
    rows = []
    for fname in os.listdir(indir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        json_path = os.path.join(indir, f"{stem}.json")
        txt_path = os.path.join(indir, f"{stem}.txt")
        if not os.path.exists(txt_path):
            continue

        # метапризнаки
        mf = MetaFeatures.load_from_json(json_path)
        df_row = mf.to_dataframe()
        df_row["index"] = mf.index

        # качества (словарь algo -> score)
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            perf = eval(content, {"__builtins__": {}})
        except Exception:
            perf = json.loads(content)

        best_algo = max(perf.items(), key=lambda kv: kv[1])[0] if isinstance(perf, dict) and perf else None
        df_row["meta__best_algo"] = str(best_algo) if best_algo is not None else None

        # опционально добавим исходные качества как perf__*
        for k, v in (perf or {}).items():
            df_row[f"perf__{k}"] = float(v)

        rows.append(df_row)

    if not rows:
        raise RuntimeError("No meta rows found in directory")

    return pd.concat(rows, ignore_index=True)  # [web:274]

def dataframe_to_dataset(df_meta: pd.DataFrame) -> Dataset:
    """
    Превращает мета-набор (строки — датасеты, столбцы — метапризнаки) в Dataset
    для дальнейшего извлечения мета-признаков: целевую колонку делаем 'target'.
    """
    # Целевую переменную возьмём как категориальную метку лучшего алгоритма,
    # чтобы StructuredMetaFeat смог посчитать свои признаки.
    if "meta__best_algo" not in df_meta.columns:
        raise ValueError("meta__best_algo column is missing in meta dataframe")

    # Копия и переименование target
    df = df_meta.copy()
    df.rename(columns={"meta__best_algo": "target"}, inplace=True)

    # Уберём явно не используемые для признаков служебные колонки (например, исходные perf__*)
    feature_cols = [c for c in df.columns if c not in ("target", "index") and not c.startswith("perf__")]
    cols_order = feature_cols + ["target"]
    df = df.loc[:, cols_order]

    # categorical_indicator: все признаки числовые, target категориальная
    meta = {
        "source": "meta_of_meta",
        "default_target_attribute": "target",
        "target_used": "target",
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "categorical_indicator": [False] * len(feature_cols),
        "attribute_names": feature_cols,
        "qualities": {
            "NumberOfInstances": int(df.shape[0]),
            "NumberOfFeatures": int(len(feature_cols) + 1),  # + target
            # target — классификация по лучшему алгоритму
            "NumberOfClasses": int(pd.Series(df["target"]).nunique(dropna=True)),
            "format": "CSV",
            "status": "active",
        },
    }
    return Dataset(meta=meta, df=df, index=-1)

def impute_numeric_columns(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    """
    Импутирует числовые колонки средним по столбцу, исключая заданные имена.
    """
    num_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "if"]
    if not num_cols:
        return df
    imputer = SimpleImputer(strategy="mean")  # [web:256]
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def main():
    ap = argparse.ArgumentParser(description="Extract meta-features from the meta-dataset (meta-of-meta).")
    ap.add_argument("indir", type=str, help="Directory with i.json (MetaFeatures) and i.txt (quality)")
    ap.add_argument("outfile", type=str, help="Path to save meta-of-meta JSON")
    args = ap.parse_args()

    # 1) Загружаем мета-набор и иммутируем пропуски в числовых колонках
    df_meta = load_meta_rows(args.indir)
    df_meta = impute_numeric_columns(df_meta, exclude=["index", "meta__best_algo"])

    # 2) Строим Dataset (features + target='meta__best_algo') из мета-набора
    meta_dataset = dataframe_to_dataset(df_meta)

    # 3) Извлекаем мета-признаки поверх мета-набора (base/stat/struct)
    base_feat = BaseMetaFeat.calculate(meta_dataset)
    stat_feat = StatisticMetaFeat.calculate(meta_dataset)
    struct_feat = StructuredMetaFeat.calculate(meta_dataset)

    meta_of_meta = MetaFeatures(
        index=-1,
        base_feat=base_feat,
        stat_feat=stat_feat,
        struct_feat=struct_feat,
    )

    # 4) Сохраняем в JSON
    with open(args.outfile, "w", encoding="utf-8") as f_out:
        json.dump(
            dataclasses.asdict(meta_of_meta),
            f_out,
            ensure_ascii=False,
            indent=2,
            ignore_nan=True,   # ключ: NaN/Inf -> null
        )

    logger.info(f"Saved meta-of-meta to {args.outfile}")

if __name__ == "__main__":
    import dataclasses  # локальный импорт для asdict
    main()
