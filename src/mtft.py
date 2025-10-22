# MeTaFeaTure module

import argparse
import dataclasses
import os
import simplejson as json
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

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


@dataclass
class MetaFeatures:
    index: int
    base_feat: BaseMetaFeat
    stat_feat: StatisticMetaFeat
    struct_feat: StructuredMetaFeat

    def save_json(self, path='data.json'):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(dataclasses.asdict(self), file, ensure_ascii=False, indent=2, ignore_nan=True)


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
            dataset = self._load_dataset_from_csv(dataset)

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
