from typing import Dict
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from src.data.dataset import Dataset
from src.utils.func import get_numeric_frame, standardize_inplace

def get_tree_features(dataset: Dataset, target_col: str = "target", max_rows: int = 10000) -> Dict[str, float]:
    X, y = get_numeric_frame(dataset, target_col=target_col)
    if X.empty or y.nunique(dropna=True) < 2:
        return {
            "depth": np.nan,
            "n_nodes": np.nan,
            "n_leaves": np.nan,
            "leaf_samples_mean": np.nan,
            "leaf_samples_std": np.nan,
            "train_acc": np.nan,
        }

    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), size=max_rows, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y
    Z = standardize_inplace(Xs.copy())
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
    try:
        clf.fit(Z, ys)
        depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()
        n_nodes = clf.tree_.node_count
        leaf_ids = clf.apply(Z)
        _, counts = np.unique(leaf_ids, return_counts=True)
        leaf_mean = float(np.mean(counts)) if counts.size else np.nan
        leaf_std = float(np.std(counts)) if counts.size else np.nan
        train_acc = float(clf.score(Z, ys))
    except Exception:
        depth = n_leaves = n_nodes = leaf_mean = leaf_std = train_acc = np.nan
    return {
        "depth": depth,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "leaf_samples_mean": leaf_mean,
        "leaf_samples_std": leaf_std,
        "train_acc": train_acc,
}


def get_knn_features(dataset: Dataset, target_col: str = "target", k: int = 3, max_rows: int = 10000) -> Dict[str, float]:
    X, y = get_numeric_frame(dataset, target_col=target_col)
    if X.empty or y.nunique(dropna=True) < 2:
        return {
            "inter_centroid_min": np.nan,
            "inter_centroid_mean": np.nan,
            "inter_centroid_max": np.nan,
            "within_dispersion_mean": np.nan,
            "separation_ratio": np.nan,
            "train_acc": np.nan,
        }
    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), size=max_rows, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y
    Z = standardize_inplace(Xs.copy())

    # Центроиды классов и межцентроидные расстояния
    centroids = Z.groupby(ys).mean(numeric_only=True)
    C = centroids.values
    if len(C) >= 2:
        # Попарные L2 расстояния между центроидами
        dists = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        dists = dists[np.triu_indices_from(dists, k=1)]
        inter_min = float(np.min(dists))
        inter_mean = float(np.mean(dists))
        inter_max = float(np.max(dists))
    else:
        inter_min = inter_mean = inter_max = np.nan

    # Внутриклассовая дисперсия (средняя L2 до центроида)
    within_vals = []
    for cls_i, grp in Z.groupby(ys):
        c = centroids.loc[cls_i].values
        dd = np.sqrt(((grp.values - c) ** 2).sum(axis=1))
        if dd.size:
            within_vals.append(float(np.mean(dd)))
    within_mean = float(np.mean(within_vals)) if within_vals else np.nan

    sep_ratio = float(inter_mean / (within_mean + 1e-12)) if np.isfinite(inter_mean) and np.isfinite(within_mean) else np.nan

    # Быстрая оценка kNN
    try:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Z, ys)
        knn_acc = float(knn.score(Z, ys))
    except Exception:
        knn_acc = np.nan

    return {
        "inter_centroid_min": inter_min,
        "inter_centroid_mean": inter_mean,
        "inter_centroid_max": inter_max,
        "within_dispersion_mean": within_mean,
        "separation_ratio": sep_ratio,
        "train_acc": knn_acc,
}


def get_linear_features(dataset: Dataset, target_col: str = "target", max_rows: int = 10000) -> Dict[str, float]:
    X, y = get_numeric_frame(dataset, target_col=target_col)
    if X.empty or y.nunique(dropna=True) < 2:
        return {
            "coef_l2_norm": np.nan,
            "coef_mean_abs": np.nan,
            "coef_sparsity_frac": np.nan,
            "intercept_abs": np.nan,
            "train_acc": np.nan,
        }
    if len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), size=max_rows, replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
    else:
        Xs, ys = X, y
    Z = standardize_inplace(Xs.copy())

    # Многокласс: one-vs-rest по умолчанию у LogisticRegression
    try:
        lr = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=1)
        lr.fit(Z, ys)
        coef = lr.coef_.ravel()
        coef_l2 = float(np.linalg.norm(coef))
        coef_mean_abs = float(np.mean(np.abs(coef)))
        coef_sparsity = float(np.mean(np.abs(coef) < 1e-6))
        intercept_abs = float(np.mean(np.abs(lr.intercept_)))
        train_acc = float(lr.score(Z, ys))
    except Exception:
        coef_l2 = coef_mean_abs = coef_sparsity = intercept_abs = train_acc = np.nan

    return {
        "coef_l2_norm": coef_l2,
        "coef_mean_abs": coef_mean_abs,
        "coef_sparsity_frac": coef_sparsity,
        "intercept_abs": intercept_abs,
        "train_acc": train_acc,
}
