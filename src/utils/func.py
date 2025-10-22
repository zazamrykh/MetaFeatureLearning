
from typing import Dict, List, Tuple

import pandas as pd
from src.data.dataset import Dataset

def get_feature_mapping(dataset : Dataset, target_col: str = "target") -> Dict:
    df = dataset.df
    feature_cols = [c for c in df.columns if c != target_col]

    cat_ind = dataset.meta.get("categorical_indicator", None)
    if cat_ind is not None and len(cat_ind) == len(feature_cols):
        is_cat_map = dict(zip(feature_cols, map(bool, cat_ind)))
    else:
        from pandas.api.types import is_categorical_dtype, is_object_dtype
        is_cat_map = {
            c: (is_object_dtype(df[c]) or is_categorical_dtype(df[c].dtype))
            for c in feature_cols
        }

    return is_cat_map


def get_numeric_series(dataset : Dataset, target_col: str = "target"):
    feature_cols = [c for c in dataset.df.columns if c != target_col]
    is_cat_map = get_feature_mapping(dataset, target_col=target_col)

    # Transform numerical columns and throw away categorical
    numeric_series: List[Tuple[str, pd.Series]] = []
    for c in feature_cols:
        if is_cat_map.get(c, False):
            continue
        s = pd.to_numeric(dataset.df[c], errors="coerce")
        numeric_series.append((c, s))

    return numeric_series


def get_numeric_frame(dataset: Dataset, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    # Берём только числовые признаки; строки приводим к числу с NaN через to_numeric
    is_cat_map = get_feature_mapping(dataset, target_col=target_col)
    feats = [c for c in dataset.df.columns if c != target_col and not is_cat_map.get(c, False)]
    X = dataset.df[feats].apply(pd.to_numeric, errors="coerce")
    y = dataset.df[target_col]
    return X, y


def standardize_inplace(X: pd.DataFrame) -> pd.DataFrame:
    # Простейшая стандартизация с защитой от нулевой дисперсии
    mu = X.mean(numeric_only=True)
    sigma = X.std(numeric_only=True).replace(0.0, 1.0)
    Z = (X - mu) / sigma
    return Z
