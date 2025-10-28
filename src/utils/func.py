
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
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


def standardize_inplace(X):
    """
    Стандартизует по столбцам (2D) или по всей оси (1D).
    - Вход: DataFrame | ndarray | sparse
    - Выход: сохраняет тип DataFrame, иначе ndarray.
    - NaN устойчивость: np.nanmean / np.nanstd
    - Нулевая дисперсия: деление на 1 вместо 0
    """
    was_df = isinstance(X, pd.DataFrame)
    if sparse.issparse(X):
        X = X.toarray()
        was_df = False

    if was_df:
        cols = X.columns
        idx = X.index
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)

    if arr.ndim == 1:
        mu = float(np.nanmean(arr))
        sigma = float(np.nanstd(arr, ddof=0))
        if sigma == 0.0 or not np.isfinite(sigma):
            sigma = 1.0
        Z = (arr - mu) / sigma
        return pd.Series(Z, index=idx, name=X.name) if was_df else Z
    else:
        # 2D: стандартизация по столбцам
        mu = np.nanmean(arr, axis=0)
        sigma = np.nanstd(arr, axis=0, ddof=0)
        sigma = np.where((sigma == 0.0) | ~np.isfinite(sigma), 1.0, sigma)
        Z = (arr - mu) / sigma
        return pd.DataFrame(Z, columns=cols, index=idx) if was_df else Z


def one_hot_encode_df(df):
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.fit(df)
    df = enc.transform(df)
    return df, enc
