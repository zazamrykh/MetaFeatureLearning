# MeTaFeaTure module

import argparse
import dataclasses
import os
import simplejson as json
import pandas as pd
import numpy as np

from dataclasses import dataclass, field, asdict, is_dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.utils.func import get_feature_mapping, get_numeric_series
from src.utils.logger import get_logger
from src.struct import get_tree_features, get_linear_features, get_knn_features
from src.data.dataset import Dataset

logger = get_logger()


class MetaFeat(ABC):
    @staticmethod
    @abstractmethod
    def calculate():
        pass

    @staticmethod
    def init_from_dict():
        pass


@dataclass
class BaseMetaFeat(MetaFeat):
    n_samples: int = 0
    n_features: int = 0
    n_cat: int = 0  # Number of categoriacal features
    n_classes: int = 0  # Number of classes. 0 if regression task

    @staticmethod
    def calculate(dataset: Dataset, target_col: str = "target"):
        df = dataset.df
        n_samples = df.shape[0]
        n_features = df.shape[1]

        is_cat_map = get_feature_mapping(dataset, target_col=target_col)
        n_cat = sum([int(is_cat) for is_cat in is_cat_map.values()])

        n_classes = int(dataset.meta['qualities']['NumberOfClasses'])

        return BaseMetaFeat(n_samples, n_features, n_cat, n_classes)

    @staticmethod
    def init_from_dict(dict):
        return BaseMetaFeat(
            n_samples = dict.get("n_samples", 0),
            n_features = dict.get("n_features", 0),
            n_cat = dict.get("n_cat", 0),
            n_classes = dict.get("n_classes", 0),
        )


statistic_metafeatures_funcs_list: List[Tuple[str, Callable[[pd.Series], float]]] = [
    ("mean", lambda s: s.mean(skipna=True)),
    ("std",  lambda s: s.std(skipna=True)),
    ("min",  lambda s: s.min(skipna=True)),
    ("max",  lambda s: s.max(skipna=True)),
]
@dataclass
class StatisticMetaFeat(MetaFeat):
    values: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def calculate(
        dataset: "Dataset",
        funcs: Optional[List[Tuple[str, Callable[[pd.Series], float]]]] = None,
        target_col: str = "target",  # Target is name by default
    ) -> "StatisticMetaFeat":
        """
        Двойная агрегация: для каждого (f1, f2) считаем f2 по вектору значений f1,
        где f1 считается по каждому числовому столбцу (категориальные пропускаются).
        """
        if funcs is None:
            funcs = statistic_metafeatures_funcs_list

        # Transform numerical columns and throw away categorical
        numeric_series: List[Tuple[str, pd.Series]] = get_numeric_series(dataset, target_col=target_col)
        results: Dict[str, float] = {}

        # Double aggregation
        for name1, f1 in funcs:
            col_vals: List[float] = []
            for _, s in numeric_series:
                v = f1(s)
                if pd.notna(v):
                    col_vals.append(float(v))
            series_vals = pd.Series(col_vals, dtype="float64")

            for name2, f2 in funcs:
                meta_name = f"{name2}_of_{name1}_per_col"
                results[meta_name] = float(f2(series_vals))

        return StatisticMetaFeat(values=results)

    @staticmethod
    def init_from_dict(dict):
        return StatisticMetaFeat(values=dict)


structured_funcs: List[Tuple[str, Callable[[Dataset], Dict[str, float]]]] = [
    ("tree",   lambda dataset: get_tree_features(dataset)),
    ("knn",    lambda dataset: get_knn_features(dataset)),
    ("linear", lambda dataset: get_linear_features(dataset)),
]
@dataclass
class StructuredMetaFeat(MetaFeat):
    values: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def calculate(
        dataset: Dataset,
        funcs: Optional[List[Tuple[str, Callable[[Dataset], Dict[str, float]]]]] = None,
        target_col: str = "target",
    ) -> "StructuredMetaFeat":
        if funcs is None:
            funcs = structured_funcs
        out: Dict[str, float] = {}
        for name, fn in funcs:
            feats = fn(dataset)
            for k, v in feats.items():
                out[f"{name}__{k}"] = float(v) if v is not None and np.isfinite(v) else np.nan
        return StructuredMetaFeat(values=out)

    @staticmethod
    def init_from_dict(dict):
        return StructuredMetaFeat(values=dict)


@dataclass
class MetaFeatures:
    index: int
    base_feat: BaseMetaFeat
    stat_feat: StatisticMetaFeat
    struct_feat: StructuredMetaFeat

    def save_json(self, path='data.json'):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(dataclasses.asdict(self), file, ensure_ascii=False, indent=2, ignore_nan=True)

    @staticmethod
    def load_from_json(path='data.json') -> "MetaFeatures":
        with open(path) as f_in:
            dict = json.load(f_in)

            return MetaFeatures(
                base_feat = BaseMetaFeat.init_from_dict(dict['base_feat']),
                stat_feat = StatisticMetaFeat.init_from_dict(dict['stat_feat']),
                struct_feat = StructuredMetaFeat.init_from_dict(dict['struct_feat']),
            )

    Number = Union[int, float, np.floating]
    MetaUnion = Union["BaseMetaFeat", "StatisticMetaFeat", "StructuredMetaFeat"]

    @staticmethod
    def _isclose(a: float, b: float, rtol: float, atol: float, nan_equal: bool) -> bool:
        # np.isclose корректно обрабатывает допуски и позволяет считать NaN равными при equal_nan=True
        return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=nan_equal))

    @staticmethod
    def _compare_numeric(v1: Any, v2: Any, rtol: float, atol: float, nan_equal: bool) -> bool:
        # Приводим numpy-скаляры и python float к float; NaN/Inf учитываются isclose
        try:
            f1 = float(v1)
            f2 = float(v2)
            return MetaFeatures._isclose(f1, f2, rtol=rtol, atol=atol, nan_equal=nan_equal)
        except Exception:
            return v1 == v2  # на случай нечисловых типов

    @staticmethod
    def _compare_value_dicts(d1: Dict[str, Any], d2: Dict[str, Any], rtol: float, atol: float, nan_equal: bool) -> Tuple[bool, Dict[str, Tuple[Any, Any]]]:
        diff: Dict[str, Tuple[Any, Any]] = {}
        if set(d1.keys()) != set(d2.keys()):
            # фиксируем отсутствующие/лишние ключи
            all_keys = set(d1.keys()) | set(d2.keys())
            for k in sorted(all_keys):
                if k not in d1 or k not in d2:
                    diff[k] = (d1.get(k, "<missing>"), d2.get(k, "<missing>"))
            return False, diff
        ok = True
        for k in d1.keys():
            v1, v2 = d1[k], d2[k]
            equal = MetaFeatures._compare_numeric(v1, v2, rtol=rtol, atol=atol, nan_equal=nan_equal)
            if not equal:
                ok = False
                diff[k] = (v1, v2)
        return ok, diff

    @staticmethod
    def compare_metafeat(a: MetaUnion, b: MetaUnion, rtol: float = 1e-6, atol: float = 1e-9, nan_equal: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Сравнивает два объекта метапризнаков:
        - BaseMetaFeat: точное сравнение целочисленных полей.
        - StatisticMetaFeat / StructuredMetaFeat: сравнение словарей значений по ключам с допусками для float и equal_nan=True.
        Возвращает (equal, report), где report содержит расхождения по полям/ключам.
        """
        # Позволяем передавать dataclass-объекты или их словари
        def as_plain(obj: Any) -> Any:
            if is_dataclass(obj):
                return asdict(obj)
            return obj

        A = as_plain(a)
        B = as_plain(b)

        # Определяем тип по наличию полей
        # Base: {n_samples, n_features, n_cat, n_classes}
        if all(k in A for k in ("n_samples", "n_features", "n_cat", "n_classes")) and \
        all(k in B for k in ("n_samples", "n_features", "n_cat", "n_classes")):
            report = {}
            ok = True
            for k in ("n_samples", "n_features", "n_cat", "n_classes"):
                if A[k] != B[k]:
                    ok = False
                    report[k] = (A[k], B[k])
            return ok, report

        # Statistic/Structured: {"values": {...}}
        if "values" in A and "values" in B and isinstance(A["values"], dict) and isinstance(B["values"], dict):
            ok, diff = MetaFeatures._compare_value_dicts(A["values"], B["values"], rtol=rtol, atol=atol, nan_equal=nan_equal)
            return ok, {"values": diff} if not ok else (True, {})

        # Fallback: глубокое сравнение словарей с допусками для чисел
        ok, diff = MetaFeatures._compare_value_dicts(A, B, rtol=rtol, atol=atol, nan_equal=nan_equal)
        return ok, diff


class MetaFeatExtractor:
    def __init__(self):
        pass

    def extract_all(self, path, save_path = 'data/extracted'):
        dataset_list = []

        logger.info(f'Extracting metafeatures from all datasets from path: {path}')
        logger.info('Loading all datasets')
        for f in os.listdir(path):
            if not f.endswith('.csv'):
                continue

            full_path = os.path.join(path, f)
            index = int(f.split('.')[0])
            df = pd.read_csv(full_path)

            try:
                with open(os.path.join(path, str(index) + '.meta.json')) as f:
                    meta = json.load(f)
            except Exception as e:
                logger.error(f'Cannot load meta for dataset with index {index}')
                meta = {}

            dataset_list.append(Dataset(meta, df, index))

        when_log = np.cumsum(np.arange(np.sqrt(len(dataset_list))))
        logger.info(f'Start extracting metafeatures')
        metafeatures_list = []
        for i, dataset in enumerate(dataset_list):
            if i in when_log:
                logger.info(f'Extracting metafeatures from dataset {i}')
            metafeatures_list.append(self.extract(dataset))

        logger.info('Finish extracting metafeatures')


        if save_path is not None:
            logger.info(f'Save metafeatures to {save_path}')

            os.makedirs(save_path, exist_ok=True)
            for metafeature in metafeatures_list:
                metafeature.save_json(os.path.join(save_path, str(metafeature.index) + '.json'))

        return metafeatures_list


    def extract(self, dataset: str | Dataset) -> MetaFeat:
        if isinstance(dataset, str):  ## May be path to csv
            dataset = Dataset.load_from_csv(dataset)

        logger.info(f'Extracting features for dataset {dataset.index}')

        logger.info(f'Extracting base features')
        base_feat = BaseMetaFeat.calculate(dataset)

        logger.info(f'Extracting stat features')
        stat_feat = StatisticMetaFeat.calculate(dataset)

        logger.info(f'Extracting struct features')
        struct_feat = StructuredMetaFeat.calculate(dataset)

        return MetaFeatures(
            index = dataset.index,
            base_feat = base_feat,
            stat_feat = stat_feat,
            struct_feat = struct_feat
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts metafeatures from datasets in source dir to target dir')
    parser.add_argument('indir', type=str, help='Path to datasets')
    parser.add_argument('outdir', type=str, help='Path to result folder')
    args = parser.parse_args()


    extractor = MetaFeatExtractor()
    extractor.extract_all(args.indir, args.outdir)
