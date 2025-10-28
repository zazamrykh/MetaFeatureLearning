#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from src.data.dataset import Dataset
from src.train import Trainer
from src.mtft import MetaFeatures
from src.utils.logger import get_logger

logger = get_logger()

def load_ready_metafeature(json_path: str) -> MetaFeatures:
    # Загружаем уже готовый JSON метапризнаков
    return MetaFeatures.load_from_json(json_path)

def main():
    parser = argparse.ArgumentParser(
        description="Load precomputed metafeatures, train/validate algorithms per dataset, and save results."
    )
    parser.add_argument("data_dir", type=str, help="Path to original datasets (.csv and .meta.json)")
    parser.add_argument("mf_dir", type=str, help="Path to precomputed metafeatures JSONs (e.g., data/pymfe_metafeatures)")
    parser.add_argument("outdir", type=str, help="Path to result folder (e.g., data/pymfe_metalearn)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random state for splitting")
    args = parser.parse_args()

    # Набор алгоритмов (3, как у тебя)
    algorithms = ["tree", "linear", "knn"]
    regr_algorithms = {
        "tree": DecisionTreeRegressor,
        "linear": LinearRegression,
        "knn": KNeighborsRegressor,
    }
    clf_algorithms = {
        "tree": DecisionTreeClassifier,
        "linear": LogisticRegression,
        "knn": KNeighborsClassifier,
    }

    os.makedirs(args.outdir, exist_ok=True)

    # Сопоставим по stem JSON метапризнаков и исходный датасет
    mf_files = [f for f in os.listdir(args.mf_dir) if f.endswith(".json")]
    if not mf_files:
        logger.error(f"No metafeature JSONs found in {args.mf_dir}")
        return

    for f in sorted(mf_files):
        stem = f[:-5]
        mf_path = os.path.join(args.mf_dir, f)
        csv_path = os.path.join(args.data_dir, f"{stem}.csv")
        meta_path = os.path.join(args.data_dir, f"{stem}.meta.json")

        if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
            logger.info(f"Skip {stem}: dataset files not found ({csv_path} / {meta_path})")
            continue

        # 1) Загружаем готовые метапризнаки
        try:
            metafeat = load_ready_metafeature(mf_path)
        except Exception as e:
            logger.info(f"Skip {stem}: cannot load metafeatures ({e})")
            continue

        # 2) Загружаем исходный датасет
        dataset = Dataset.load(csv_path, meta_path, int(stem) if stem.isdigit() else -1)
        logger.info(f"Train algorithms for dataset: {dataset.index}")

        # 3) Выбор режима задачи по метапризнакам
        is_classification = (int(getattr(metafeat.base_feat, "n_classes", 0)) != 0)

        # 4) Разбиение
        try:
            train_df, test_df = train_test_split(dataset.df, test_size=args.test_size, random_state=args.seed, shuffle=True)  # [web:248]
        except Exception as e:
            logger.info(f"Skip {stem}: split failed ({e})")
            continue

        train_dataset = Dataset(dataset.meta, train_df, dataset.index)
        test_dataset = Dataset(dataset.meta, test_df, dataset.index)

        # 5) Обучение и оценка
        quality_on_algorithms = {}
        for algo in algorithms:
            try:
                if is_classification:
                    Model = clf_algorithms[algo]
                    # Логистической регрессии часто нужны параметры по умолчанию
                    model = Model()
                else:
                    Model = regr_algorithms[algo]
                    model = Model()

                trainer = Trainer(model=model, is_classification=is_classification)
                trainer.fit(train_dataset)
                result = trainer.eval(test_dataset)
                quality_on_algorithms[algo] = result
            except Exception as e:
                logger.info(f"{stem} - {algo}: failed ({e})")
                quality_on_algorithms[algo] = float("nan")

        # 6) Сохранение: метапризнаки и качества
        # Сохраняем исходные (загруженные) метапризнаки без изменений, как в прежнем пайплайне
        out_json = os.path.join(args.outdir, f"{stem}.json")
        metafeat.save_json(path=out_json)

        out_txt = os.path.join(args.outdir, f"{stem}.txt")
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(str(quality_on_algorithms))

if __name__ == "__main__":
    main()
