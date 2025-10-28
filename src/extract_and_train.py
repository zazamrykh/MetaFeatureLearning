import argparse
import os

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import train_test_split

from src.data.dataset import Dataset
from src.train import Trainer
from src.mtft import MetaFeatExtractor
from src.utils.logger import get_logger

logger = get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts metafeatures from datasets, train ml algorightms for them and save metafeatures with result metrics')
    parser.add_argument('indir', type=str, help='Path to datasets')
    parser.add_argument('outdir', type=str, help='Path to result folder')
    args = parser.parse_args()

    extractor = MetaFeatExtractor()
    metafeatures, datasets = extractor.extract_all(args.indir, save_path =None, return_datasets=True)

    algorithms = ['tree', 'linear', 'knn']
    regr_algorithms = {'tree': DecisionTreeRegressor, 'linear': LinearRegression, 'knn': KNeighborsRegressor}
    clf_algorithms = {'tree': DecisionTreeClassifier, 'linear': LogisticRegression, 'knn': KNeighborsClassifier}
    quality_on_datasets_list = []
    for metafeat, dataset in zip(metafeatures, datasets):
        logger.info(f'Train algorithms for dataset: {dataset.index}')
        quality_on_algorithms_dict = {}
        for algo in algorithms:
            if metafeat.base_feat.n_classes == 0:
                is_classification = False
                model = regr_algorithms[algo]()
            else:
                is_classification = True
                model = clf_algorithms[algo]()

            train, test = train_test_split(dataset.df, test_size=0.2)
            train_dataset = Dataset(dataset.meta, train, dataset.index)
            test_dataset = Dataset(dataset.meta, test, dataset.index)

            trainer = Trainer(model=model, is_classification=is_classification)
            trainer.fit(train_dataset)
            result = trainer.eval(test_dataset)

            quality_on_algorithms_dict[algo] = result

        quality_on_datasets_list.append(quality_on_algorithms_dict)

    os.makedirs(args.outdir, exist_ok=True)
    for i, (metafeat, quality) in enumerate(zip(metafeatures, quality_on_datasets_list)):
        metafeat.save_json(path=os.path.join(args.outdir, f'{i}.json'))

        with open(os.path.join(args.outdir, f'{i}.txt'), mode='w') as f:
            f.write(str(quality))
