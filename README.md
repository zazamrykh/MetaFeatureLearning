### Data loading

You can load datasets using openml with src/data/load_datasets.py
Run as follows for downloading 300 datasets into data directory:
  `python3 -m src.data.load_datasets --output-dir data --n-datasets 300`

### Metafeatures extraction
You can extract base, structural and statistical metafeatures for dataset something like that:
  `python3 -m src.mtft data data/metafeatures`

### Invariance test
Shuffle dataframe, permute columns and reorder cat features running that script:
  `python3 -m src.exp.invariance_test`

### Train metaalgorithm
Train metaalgorithm for model selection. Also compare with naive approach (take only best statistical alorithm)
  `python3 -m src.exp.train_meta data/metalearn`
Logs will be available in logs/exps directory

### Visualization
You can use visualization of result metafeatures where label is best model which solve task the best (implemented for tree, knn and linear) as follows:
  `python3 -m src.exp.visualize_metaclasses data/metafeatures --method umap --save visualization/umap.png`

### Get meta from meta
Special script for obtaining metafeatures for meta dataset:
  `python3 -m src.exp.meta_from_meta data/metalearn data/meta_from_meta.json`

### PyMFE
Use pymfe to repeat metafeatures extraction, invariance test, visualization and train algorithm
  `python3 -m src.pymfe_extract`
  `python3 -m src.exp.invariance_test_pymfe data/datasets/26.csv data/datasets/26.meta.json`
  `python3 -m src.get_quality  data/datasets data/pymfe_metafeatures data/pymfe_metalearn`
  `python3 -m src.exp.train_meta data/pymfe_metalearn`
