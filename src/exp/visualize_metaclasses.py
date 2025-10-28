#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import dataclasses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# UMAP опционален; если не установлен, fallback на PCA
from umap import UMAP
HAS_UMAP = True


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ----- Ваши классы -----
# Предполагается, что классы уже доступны из вашего проекта:
# from src.mtft import MetaFeatures, BaseMetaFeat, StatisticMetaFeat, StructuredMetaFeat
# Для самодостаточности добавим минимальные init_from_dict и to_dataframe
from dataclasses import dataclass, field

@dataclass
class BaseMetaFeat:
    n_samples: int = 0
    n_features: int = 0
    n_cat: int = 0
    n_classes: int = 0

    @staticmethod
    def init_from_dict(d):
        return BaseMetaFeat(
            n_samples=d.get("n_samples", 0),
            n_features=d.get("n_features", 0),
            n_cat=d.get("n_cat", 0),
            n_classes=d.get("n_classes", 0),
        )

@dataclass
class StatisticMetaFeat:
    values: dict = field(default_factory=dict)

    @staticmethod
    def init_from_dict(d):
        # В файлах у вас stat_feat: {"values": {...}} или сразу {...}
        if "values" in d:
            return StatisticMetaFeat(values=d["values"])
        return StatisticMetaFeat(values=d)

@dataclass
class StructuredMetaFeat:
    values: dict = field(default_factory=dict)

    @staticmethod
    def init_from_dict(d):
        if "values" in d:
            return StructuredMetaFeat(values=d["values"])
        return StructuredMetaFeat(values=d)

@dataclass
class MetaFeatures:
    index: int
    base_feat: BaseMetaFeat
    stat_feat: StatisticMetaFeat
    struct_feat: StructuredMetaFeat

    def to_dataframe(self) -> pd.DataFrame:
        row = {
            "index": int(self.index),
            "base__n_samples": int(self.base_feat.n_samples),
            "base__n_features": int(self.base_feat.n_features),
            "base__n_cat": int(self.base_feat.n_cat),
            "base__n_classes": int(self.base_feat.n_classes),
        }
        for k, v in (self.stat_feat.values or {}).items():
            row[f"stat__{k}"] = v
        for k, v in (self.struct_feat.values or {}).items():
            row[f"struct__{k}"] = v
        return pd.DataFrame([row])

    @staticmethod
    def load_from_json(path: str) -> "MetaFeatures":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return MetaFeatures(
            index=d.get("index", -1),
            base_feat=BaseMetaFeat.init_from_dict(d["base_feat"]),
            stat_feat=StatisticMetaFeat.init_from_dict(d["stat_feat"]),
            struct_feat=StructuredMetaFeat.init_from_dict(d["struct_feat"]),
        )

# ----- Утилиты загрузки и визуализации -----

def load_meta_and_quality(root_dir: str):
    """
    Читает все *.json и соответствующие *.txt (качество алгоритмов) из каталога.
    Возвращает:
      - df_feats: объединённый DataFrame метапризнаков (по строке на датасет)
      - df_labels: DataFrame c колонками алгоритмов и меткой best_algo
    """
    rows = []
    labels = []
    for fname in os.listdir(root_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]  # без .json
        json_path = os.path.join(root_dir, f"{stem}.json")
        txt_path = os.path.join(root_dir, f"{stem}.txt")
        if not os.path.exists(txt_path):
            # пропускаем, если нет файла качества
            continue

        # загрузка метапризнаков
        mf = MetaFeatures.load_from_json(json_path)
        df_row = mf.to_dataframe()
        # загрузка качества
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Ожидаем формат словаря, например: {'tree': 0.3, 'linear': 0.31, 'knn': 0.33}
        try:
            perf = eval(content, {"__builtins__": {}})
        except Exception:
            # Попробуем JSON, если там json-словарь
            try:
                perf = json.loads(content)
            except Exception:
                continue

        # определяем лучший алгоритм
        if isinstance(perf, dict) and len(perf) > 0:
            best_algo = max(perf.items(), key=lambda kv: kv[1])[0]
        else:
            best_algo = None

        df_row["meta__best_algo"] = best_algo
        # сохраним также исходные качества как отдельные колонки
        for k, v in perf.items():
            df_row[f"perf__{k}"] = v

        rows.append(df_row)
        labels.append((mf.index, best_algo, perf))

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df_feats = pd.concat(rows, ignore_index=True)
    # Сформируем таблицу меток и качеств
    label_rows = []
    for idx, best, perf in labels:
        r = {"index": idx, "best_algo": best}
        for k, v in perf.items():
            r[k] = v
        label_rows.append(r)
    df_labels = pd.DataFrame(label_rows)
    return df_feats, df_labels

def reduce_to_2d(X: np.ndarray, method: str = "umap", random_state: int = 42):
    """
    Проецирует признаки в 2D:
      - umap (если установлен)
      - иначе t-SNE (медленнее)
      - иначе PCA
    Перед проекцией выполняется стандартизация.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if method == "umap" and HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=random_state)
        print('UMAP')
        Z = reducer.fit_transform(Xs)
        return Z, "UMAP"
    if method == "tsne":
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        Z = tsne.fit_transform(Xs)
        return Z, "t-SNE"
    # fallback
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(Xs)
    return Z, "PCA"

def plot_embedding(Z: np.ndarray, labels: pd.Series, title: str = "Meta-datasets 2D projection", save_path: str | None = None):
    plt.figure(figsize=(8, 6))
    # Цвета по мета-классу
    classes = labels.unique()
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = labels == cls
        color = cmap(i % 10)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=30, c=[color], label=str(cls), alpha=0.8, edgecolors="none")
    plt.legend(title="Best algo", loc="best")
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Visualize meta-datasets in 2D with colors by best algorithm.")
    ap.add_argument("indir", type=str, help="Directory with i.json (metafeatures) and i.txt (qualities)")
    ap.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca"], help="Dimensionality reduction method")
    ap.add_argument("--save", type=str, default="", help="Path to save the plot (.png). If empty, show interactively.")
    args = ap.parse_args()

    df_feats, df_labels = load_meta_and_quality(args.indir)
    if df_feats.empty:
        print("No meta files found.")
        return

    # Собираем матрицу признаков: все числовые колонки, исключаем служебные
    drop_cols = ["index", "meta__best_algo"]
    num_cols = [c for c in df_feats.columns if c not in drop_cols and df_feats[c].dtype.kind in "if"]
    X = df_feats[num_cols].to_numpy()

    # Импутация NaN медианами по столбцам (устойчивость)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # 2D проекция
    Z, used = reduce_to_2d(X_imp, method=args.method)

    # Вектор мета-классов
    y_meta = df_feats["meta__best_algo"].astype(str)

    # Рисуем
    title = f"Meta-datasets: 2D projection by {used}"
    save_path = args.save if args.save else None
    plot_embedding(Z, y_meta, title=title, save_path=save_path)

if __name__ == "__main__":
    main()
