from dataclasses import dataclass
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype

@dataclass
class Dataset:
    meta: Dict
    df: pd.DataFrame
    index: int

    @staticmethod
    def load(df_path, meta_path, index):
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(df_path)
        return Dataset(meta, df, index)

    def shuffle_dataset(self) -> None:
        # 1) Перемешиваем строки (восстанавливая компактный индекс)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        # 2) Определяем имя целевой колонки (если есть в meta)
        target_name = (
            self.meta.get("target_used")
            or self.meta.get("default_target_attribute")
        )
        cols = list(self.df.columns)
        feature_cols: List[str] = [c for c in cols if c != target_name] if target_name in cols else cols[:]

        # 3) Переставляем столбцы признаков (таргет оставляем в конце, если он есть)
        perm_features = list(np.random.permutation(feature_cols))
        new_order = perm_features + ([target_name] if target_name in cols else [])
        self.df = self.df.loc[:, new_order]

        # 4) Обновляем meta.attribute_names и meta.categorical_indicator под новую перестановку
        #    Берём исходный порядок признаков из meta, если он есть; иначе — текущий feature_cols
        orig_attr_names = self.meta.get("attribute_names", feature_cols)
        orig_cat_ind = self.meta.get("categorical_indicator", None)
        # Строим карту признак -> индикатор категориальности по исходному порядку
        if isinstance(orig_cat_ind, list) and len(orig_cat_ind) == len(orig_attr_names):
            cat_map = dict(zip(orig_attr_names, map(bool, orig_cat_ind)))
            new_cat_ind = [bool(cat_map.get(c, False)) for c in perm_features]
            self.meta["categorical_indicator"] = new_cat_ind
        # Переписываем порядок имён признаков
        self.meta["attribute_names"] = perm_features

        # 5) Меняем порядок категорий в категориальных столбцах признаков
        for c in perm_features:
            s = self.df[c]
            # Используем индикатор из meta, если он есть, иначе смотрим по dtype
            is_cat = False
            if "categorical_indicator" in self.meta and len(self.meta["categorical_indicator"]) == len(perm_features):
                idx = perm_features.index(c)
                is_cat = bool(self.meta["categorical_indicator"][idx])
            else:
                is_cat = is_object_dtype(s) or is_categorical_dtype(s.dtype)

            if is_cat:
                if not is_categorical_dtype(s.dtype):
                    s = s.astype("category")
                cats = list(s.cat.categories)
                if len(cats) > 1:
                    new_cats = list(np.random.permutation(cats))
                    s = s.cat.reorder_categories(new_cats, ordered=s.cat.ordered if s.cat.ordered is not None else False)
                self.df[c] = s

        # 6) Также меняем порядок категорий у целевого столбца (если он есть и категориален)
        if target_name in self.df.columns:
            t = self.df[target_name]
            is_target_cat = is_object_dtype(t) or is_categorical_dtype(t.dtype)
            # Если явно известно число классов в meta и оно > 1, это дополнительная эвристика
            n_classes = None
            try:
                n_classes = int(self.meta.get("qualities", {}).get("NumberOfClasses", 0))
            except Exception:
                n_classes = None
            if is_target_cat or (n_classes is not None and n_classes > 1):
                if not is_categorical_dtype(t.dtype):
                    t = t.astype("category")
                cats_t = list(t.cat.categories)
                if len(cats_t) > 1:
                    new_cats_t = list(np.random.permutation(cats_t))
                    t = t.cat.reorder_categories(new_cats_t, ordered=t.cat.ordered if t.cat.ordered is not None else False)
                self.df[target_name] = t
